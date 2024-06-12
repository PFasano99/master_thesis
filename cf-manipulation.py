import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from multiprocessing.pool import ThreadPool
import json
import tqdm
import pickle
import random

def find_all_extention_files(path_to_file = "saved-feat", save_dir = "datasets/cnr/hl2-sensor-dump-i32-cnr/colored_features", extension = '.pt'):
    """
        Returns the list of all the file names for a given extention and checks if the save_dir path exists and creates the folder if need be.

        Input:
            - path_to_file:str the path to the .pt files to read
            - save_dir:str the path of the forlder to save the generated images
            - extension:str the extension to look for in the path_to_file
        Retun: 
            - files_extension all the file names where estension is the one given  
    """
    if not os.path.exists(path_to_file):
        print("The path: ", path_to_file, " does not exist!\n Check the path in input")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files_extension = [f for f in os.listdir(path_to_file) if f.endswith(extension)]

    if len(files_extension) == 0:
        print(f"No *{extension} files where found in the folder, please check the path!")
        print("The path in input was: ", path_to_file)
        files_extension = []
    
    return files_extension


def convert_pt_to_img(path_to_file = "saved-feat", save_dir = "datasets/cnr/hl2-sensor-dump-i32-cnr/colored_features", image_height = 1080, image_width = 1920):
    """
        This method takes in input the folder where the .pt files are located 
        (files created by the extract_conceptfusion_features.py from https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py)
        and returns in the save_dir the images produced coloring the extracted features using sklearn PCA
    
        Input:
            - path_to_file:str the path to the .pt files to read
            - save_dir:str the path of the forlder to save the generated images
    """
    
    pt_files_names = find_all_extention_files(path_to_file, save_dir, extension='.pt')

    for pt_file_name in tqdm.tqdm(pt_files_names, desc = "Coloring images based on features"):
    
        pt_file = torch.load(path_to_file+"/"+pt_file_name)

        pt_file = pt_file.view(-1, pt_file.size()[2])
        pca = PCA(n_components=3)
        pca_r = pca.fit_transform(pt_file)
        pca_r = MinMaxScaler(feature_range = (0, 255), clip = True).fit_transform(pca_r)
        pca_r = pca_r.reshape((image_height, image_width, 3))

        cv2.imwrite((save_dir+"/"+pt_file_name[:-3]+".png"), pca_r)

def pt_batch_to_img(path_to_file = "saved-feat", save_dir = "datasets/cnr/hl2-sensor-dump-i32-cnr/colored_features", pt_names = [], pics_to_sample = 2, image_height = 1080, image_width = 1920):
    """
        This method takes in input the folder where the .pt files are located 
        (files created by the extract_conceptfusion_features.py from https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py)
        and returns in the save_dir the images produced coloring the extracted features using sklearn PCA

        Input:
            - path_to_file:str the path to the .pt files to read
            - save_dir:str the path of the forlder to save the generated images
            - pt_names:array:str the array of names (e.g ["133416009812234976.pt","133416009805570974.pt"]) of the .pt files to read, if empty the method will read n pictures (pics_to_sample) at random
            - pics_to_sample:int number of pics to sample if pt_names is empty
    """

    if len(pt_names) == 0:
        pt_files_names = pt_files_names = find_all_extention_files(path_to_file, save_dir, extension='.pt')

        if pics_to_sample == -1 or pics_to_sample > len(pt_files_names):
            print("Converting all pt files in input folder      tot: ", len(pt_files_names))
            pt_names = pt_files_names  
        else:
            pt_names = np.random.choice(pt_files_names, pics_to_sample, replace=False)

    pt_files = []
    for pt_file_name in tqdm.tqdm(pt_names, desc = "Reading pt files"):
        pt = torch.load(path_to_file+"/"+pt_file_name)
        pt = pt.view(-1, pt.size()[2])
        pt_files.append(pt)
    
    # Concatenate feature matrices
    all_frames_features = np.concatenate(pt_files)
    # Apply PCA to all frames
    pca = PCA(n_components=3)
    pca_r = pca.fit_transform(all_frames_features.reshape(-1, all_frames_features.shape[-1]))
    pca_r = MinMaxScaler(feature_range=(0, 255), clip=True).fit_transform(pca_r)
    # reshape the resulting pca to have the original images e.g. from (38400, 3)  to (2,19200, 3)
    pca_r = pca_r.reshape((len(pt_names),int(pca_r.shape[0]/len(pt_names)),pca_r.shape[1]))

    #per image reshape it to be desplayable as an image
    i = 0
    for pt_file_name in tqdm.tqdm(pt_names, desc = "Saving feature to png"):

        pcar_s = pca_r[i]
        pcar_s = pcar_s.reshape((image_height, image_width, 3))
        cv2.imwrite((save_dir+"/"+pt_file_name[:-3]+"_batch.png"), pcar_s)
        i+=1

from concurrent.futures import ThreadPoolExecutor

def pca_batch(path_to_file = "saved-feat", save_dir="datasets/cnr/hl2-sensor-dump-i32-cnr/colored_features", image_height=1080, image_width=1920, subset_size=1000):
    pt_files_names = find_all_extention_files(path_to_file, save_dir, extension='.pt')
    # Select a random subset of the data for PCA fitting
    sampled_pt_files_names = random.sample(pt_files_names, subset_size)
    print("sampled_pt_files_names", len(sampled_pt_files_names))
    # Combine data from all .pt files to fit the PCA
    pt_files = []
    for pt_file_name in tqdm.tqdm(sampled_pt_files_names, desc = "Reading pt files"):
        pt = torch.load(path_to_file+"/"+pt_file_name)
        pt = pt.view(-1, pt.size()[2])
        pt_files.append(pt)
    
    print("concatenating pt_files")
    # Concatenate feature matrices
    subset_data = np.concatenate(pt_files)
    # Fit PCA and scaler on the subset data
    print("performing PCA on subset")
    pca = PCA(n_components=3)
    pca.fit(subset_data)
    scaler = MinMaxScaler(feature_range=(0, 255), clip=True)
    scaler.fit(pca.transform(subset_data))

    
    for pt_file_name in tqdm.tqdm(pt_files_names, desc = "writing pca files"):
        pt_file = torch.load(os.path.join(path_to_file, pt_file_name))
        pt_file = pt_file.view(-1, pt_file.size()[2]).numpy()
        pca_r = pca.transform(pt_file)
        pca_r = scaler.transform(pca_r)
        pca_r = pca_r.reshape((image_height, image_width, 3))
        cv2.imwrite(os.path.join(save_dir, pt_file_name[:-3] + ".png"), pca_r)
    

def draw_masks_fromDict(image, masks_generated) :
  masked_image = image.copy()
  for i in range(len(masks_generated)) :
    masked_image = np.where(np.repeat(masks_generated[i]['segmentation'].astype(int)[:, :, np.newaxis], 3, axis=2),
                            np.random.choice(range(256), size=3),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)

  return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

def SAM_masks(path_to_file = "saved-feat", save_dir = "datasets/cnr/hl2-sensor-dump-i32-cnr/colored_features"):
    """
        This method takes in input the folder where the .pkl files are located 
        (files created by the extract_conceptfusion_features.py from https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py)
        and returns in the save_dir the images produced coloring the masks generated by the SAM (https://github.com/facebookresearch/segment-anything)

        Input:
            - path_to_file:str the path to the .pkl files to read
            - save_dir:str the path of the forlder to save the generated images
    """
    pkl_files_names = find_all_extention_files(path_to_file, save_dir, extension='.pkl')

    for pkl_file_name in tqdm.tqdm(pkl_files_names, desc = "Coloring images based on features"):

        with open(path_to_file+"/"+pkl_file_name, 'rb') as f:
            result_dict = pickle.load(f)
    
        # Initialize an empty canvas to store the output image
        output_image = np.zeros((result_dict[0]['segmentation'].shape[0], result_dict[0]['segmentation'].shape[1], 3), dtype=np.uint8)
        output_image = draw_masks_fromDict(output_image, result_dict)
        cv2.imwrite((save_dir+"/"+pkl_file_name[:-3]+"_mask.png"), output_image) 

#convert_pt_to_img(path_to_file = "./build_depth/dataset/cnr_c60/saved-feat" ,save_dir = "./build_depth/dataset/cnr_c60/colored_features/feature_to_rgb")
#SAM_masks(path_to_file = "./build_depth/dataset/cnr_c60/saved-feat" ,save_dir = "./build_depth/dataset/cnr_c60/colored_features/SAM_masks")
#pt_batch_to_img(path_to_file = "./datasets/cnr/hl2-sensor-dump-i32-cnr_original_images/saved-feat" ,save_dir = "datasets/cnr/hl2-sensor-dump-i32-cnr_original_images/colored_features/batch_pca", pt_names=["133416009812234976.pt","133416009805570974.pt", "133416009798573771.pt"])
#pt_batch_to_img(path_to_file = "./build_depth/dataset/cnr_c60/saved-feat" ,save_dir = "./build_depth/dataset/cnr_c60/colored_features/batch_pca", pics_to_sample= -1)
pca_batch(path_to_file = "./build_depth/dataset/cnr_c60/saved-feat-all" ,save_dir = "./build_depth/dataset/cnr_c60/colored_features/batch_pca", subset_size=10)
