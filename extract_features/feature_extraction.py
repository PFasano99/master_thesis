import os
import pickle as pkl
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

import torch
import open3d as o3d
import open_clip
from torchvision import transforms
from gradslam import *
import tyro
from gradslam.datasets import ( 
    ICLDataset,
    ReplicaDataset,
    ScannetDataset,
    load_dataset_config,
)
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image
import cv2

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm, trange
from typing_extensions import Literal

import gc, psutil, imutils


# Torch device to run computation on (E.g., "cpu")
device: str = "cpu"

# SAM checkpoint and model params
checkpoint_path: Union[str, Path] = (
    Path.home()
    / "code"
    / "gradslam-foundation"
    / "examples"
    / "checkpoints"
    / "sam_vit_h_4b8939.pth"
)
model_type = "vit_h"
# Ignore masks that have valid pixels less than this fraction (of the image area)
bbox_area_thresh: float = 0.0005
# Number of query points (grid size) to be sampled by SAM
points_per_side: int = 32

input_folder = "./resources/dataset/dump_cnr_c60/PV/"       # Path to folder containing images

# Stride (number of frames to skip between successive fusion steps)
stride: int = 10
# Desired image width and height
desired_height: int = 1080
desired_width: int = 1920

#the feature extractor model to use
feature_extractor = "openClip" #["openClip","dinoV2"]

# openclip model config
open_clip_model = "ViT-H-14"
open_clip_pretrained_dataset = "laion2b_s32b_b79k"

# dinov2 model config
dinov2_model = "facebookresearch/dinov2"
dinov2_pretrained_dataset = "dinov2_vitl14"

# Directory to save extracted features
if feature_extractor == "dinoV2":
    save_dir = "./resources/dataset/cnr_c60/dinov2-saved-feat/"  # Path to folder to save .bin files
elif feature_extractor == "openClip":
    save_dir = "./resources/dataset/cnr_c60/openClip-saved-feat/"  # Path to folder to save .bin files

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

feat_dim = 1024  
image_height = 1080
image_width = 1920

# Transformation to pad the image and normalize

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_rgb_paths(image_folder="./RV"):
    image_paths = []
    for image_name in os.listdir(image_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Load and preprocess the image
            image_path = os.path.join(image_folder, image_name)
            image_paths.append(image_path)
    
    return image_paths

def get_pkl_files(folder_path):
    return [file for file in os.listdir(folder_path) if file.endswith('.pkl')]


def get_ids_every_stride(ids, stride):
    return ids[::stride]

def main(): 

    torch.autograd.set_grad_enabled(False)
    
    image_paths = get_rgb_paths(input_folder)
    image_paths = get_ids_every_stride(image_paths, stride)
    print(len(image_paths))
    
    sam = sam_model_registry[model_type](checkpoint=Path(checkpoint_path))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    os.makedirs(save_dir, exist_ok=True)
    
    print("Extracting SAM masks...")
    
    for image_path in len(image_paths):
        img = Image.open(image_path).convert("RGB")   
        masks = mask_generator.generate(img)
        cur_mask = masks[0]["segmentation"]
        _savefile = os.path.join(
            save_dir,
            os.path.splitext(os.path.basename(image_path))[0] + ".pkl",
        )
        with open(_savefile, "wb") as f:
            pkl.dump(masks, f, protocol=pkl.HIGHEST_PROTOCOL)

    model = None
    if feature_extractor == "openClip":
        print(
            f"Initializing CLIP model: {open_clip_model}"
            f" pre-trained on {open_clip_pretrained_dataset}"
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            open_clip_model, open_clip_pretrained_dataset
        ) 

    elif feature_extractor == "dinoV2":

        print(
            f"Initializing DinoV2 model: {dinov2_model}"
            f" pre-trained on {dinov2_pretrained_dataset}..."
        )

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device)

    model.cpu()    
    model.eval()

    timestamps = get_pkl_files(save_dir)
    print("pkl file found: ", len(timestamps))
    
    print("Computing pixel-aligned features...")
    for idx in tqdm(timestamps):
        maskfile = "./resources/dataset/cnr_c60/dinov2-saved-feat/"+idx
        
        with open(maskfile, "rb") as f:
            masks = pkl.load(f)
        
        imgfile = input_folder+idx[:-3]+"png"
        img = None

        if feature_extractor == "openClip":                    
            img = cv2.imread(imgfile)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elif feature_extractor == "dinoV2":
            img = Image.open(imgfile).convert("RGB")

        LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image_height, image_width
        global_feat = None
        with torch.cuda.amp.autocast():
            if feature_extractor == "openClip":                    
                _img = preprocess(Image.open(imgfile)).unsqueeze(0)
                global_feat = model.encode_image(_img)
            elif feature_extractor == "dinoV2":
                _img = transform(img).unsqueeze(0).float()
                global_feat = model(_img)
            global_feat /= global_feat.norm(dim=-1, keepdim=True)

        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)

        feat_dim = global_feat.shape[-1]
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        
        for maskidx in range(len(masks)):
            _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])  # xywh bounding box
            nonzero_inds = torch.argwhere(torch.from_numpy(masks[maskidx]["segmentation"]))
            
            roifeat = None
            if feature_extractor == "openClip":                    
                img_roi = img[_y : _y + _h, _x : _x + _w, :]
                img_roi = Image.fromarray(img_roi)
                img_roi = preprocess(img_roi).unsqueeze(0)
                roifeat = model.encode_image(img_roi)
            elif feature_extractor == "dinoV2":
                img_roi = img.crop((_x, _y, _x + _w, _y + _h))
                img_roi = transform(img_roi).unsqueeze(0).float()
                roifeat = model(img_roi)

            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)

        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
        outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)
        for maskidx in range(len(masks)):
            _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
                outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
            ).half()

        outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for half in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(outfeat, [desired_height, desired_width], mode="nearest")
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0]  # --> H, W, feat_dim

        tensor = outfeat.detach().cpu() 
        torch_np = tensor.numpy()

        save_path = save_dir+idx[:-3]+"bin"

        print("saving at: ", save_path)
        with open(save_path, 'wb') as f:
            np.array(torch_np, dtype=np.float32).tofile(f)


if __name__ == "__main__":
    main()