import os
import torch
import json
import tqdm

import cv2
import numpy as np

import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


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

def convert_bin_to_img(path_to_file = "./resources/dataset/cnr_c60/concat_feats/", save_dir = "./resources/dataset/cnr_c60/concat_feats/rgb_feats", image_height = 1080, image_width = 1920):
    """
        This method takes in input the folder where the .pt files are located 
        (files created by the extract_conceptfusion_features.py from https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py)
        and returns in the save_dir the images produced coloring the extracted features using sklearn PCA
    
        Input:
            - path_to_file:str the path to the .pt files to read
            - save_dir:str the path of the forlder to save the generated images
    """
    
    bin_files_names = find_all_extention_files(path_to_file, save_dir, extension='.bin')

    for bin_file_name in tqdm.tqdm(bin_files_names, desc = "Coloring images based on features"):
        y = np.fromfile(path_to_file+bin_file_name, dtype=np.float32)
        y = y[:-3]  #this is here because the generated file has a 3 char eof sequence
        bin_file = torch.from_numpy(y).reshape(image_height,image_width,-1)
        bin_file = bin_file.view(-1, bin_file.size()[2])
        pca = PCA(n_components=3)
        pca_r = pca.fit_transform(bin_file)
        pca_r = MinMaxScaler(feature_range = (0, 255), clip = True).fit_transform(pca_r)
        pca_r = pca_r.reshape((image_height, image_width, 3))

        cv2.imwrite((save_dir+"/"+bin_file_name[:-4]+".png"), pca_r)



convert_bin_to_img()