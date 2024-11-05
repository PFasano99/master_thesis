import os
import torch
import json
import tqdm

import cv2
import numpy as np

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool


def find_all_extention_files(path_to_file = "./resources/dataset/cnr_c60/concat_feats/", save_dir = "./resources/dataset/cnr_c60/concat_feats/colored_features", extension = '.bin'):
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

# Function to process each bin file
def process_bin_file(bin_file_name, path_to_file= "./resources/dataset/cnr_c60/concat_feats/", save_dir= "./resources/dataset/cnr_c60/concat_feats/rgb_feats", image_height = 1080, image_width = 1920):
    print("saving: ",bin_file_name)
    bin_from_file = np.fromfile(path_to_file + bin_file_name, dtype=np.float32)
        
    bin_file = torch.from_numpy(bin_from_file).reshape(image_height, image_width, -1)
    bin_file = bin_file.view(-1, bin_file.size()[2])
    print(bin_file.shape)

    pca = PCA(n_components=3)
    rand_proj = torch.rand((bin_file.shape[1], 1))
    projected = bin_file.matmul(rand_proj)
    _, index = np.unique(projected, return_index=True)
    pca.fit(bin_file[index])

    pca_r = pca.transform(bin_file)
    pca_r = MinMaxScaler(feature_range=(0, 255), clip=True).fit_transform(pca_r)
    print(type(pca_r))
    if("all_feats.bin" in bin_file_name):
        target_height, target_width = 1450, 2000
        # Create a list of the difference in dimensions
        pad_width = (1450*2000) #- len(pca_r)

        padded_image = np.zeros((pad_width,3), dtype=np.float32)
        padded_image[:len(pca_r)] = pca_r[:len(pca_r)]
        print("len(padded_image) ", len(padded_image))
        padded_image = padded_image.reshape((target_height, target_width, 3))

        cv2.imwrite((save_dir + "/3_" + bin_file_name[:-3] + "png"), padded_image)
        
    else:
        pca_r = pca_r.reshape((image_height, image_width, 3))
        cv2.imwrite((save_dir + "/" + bin_file_name[:-3] + "png"), pca_r)

#process_bin_file("133468485619166969.bin", "./resources/dataset/cnr_c60/openClip-saved-feat/")
#process_bin_file("dinoV2_all_feats.bin", "./resources/dataset/cnr_c60/concat_feats/", image_height=2804853, image_width=1)


def convert_bin_to_img(path_to_file = "./resources/dataset/cnr_c60/concat_feats/", save_dir = "./resources/dataset/cnr_c60/concat_feats/rgb_feats", image_height = 1080, image_width = 1920, num_thread = 32):
    """
        This method takes in input the folder where the .pt files are located 
        (files created by the extract_conceptfusion_features.py from https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py)
        and returns in the save_dir the images produced coloring the extracted features using sklearn PCA
    
        Input:
            - path_to_file:str the path to the .pt files to read
            - save_dir:str the path of the forlder to save the generated images
    """
    print("reading all files with .bin extension in: ", path_to_file, "\n")
    os.makedirs(save_dir, exist_ok=True)

    bin_files_names = find_all_extention_files(path_to_file, save_dir, extension='.bin')

    for file_name in tqdm.tqdm(bin_files_names, desc=" Coloring images based on features"):
        print(file_name)
        if file_name == "all_feats.bin":
            #print("skip")
            process_bin_file(file_name, path_to_file, save_dir, 2804853, 1)
        elif "_vtx.bin" not in file_name:
            #print("skip")
            process_bin_file(file_name, path_to_file, save_dir, image_height, image_width)

#convert_bin_to_img()
#convert_bin_to_img(path_to_file = "./resources/dataset/cnr_c60/openClip_saved-feat/", save_dir = "./resources/dataset/cnr_c60/open_clip_rgb")
#convert_bin_to_img(path_to_file = "./resources/dataset/cnr_c60/dinov2-saved-feat/", save_dir = "./resources/dataset/cnr_c60/dinov2_rgb")
