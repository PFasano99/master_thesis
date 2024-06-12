import torch
import open3d as o3d
import numpy as np
import os
from gradslam.structures.pointclouds import Pointclouds
import tqdm

def save_h5_to_ply(load_path = "./build_depth/dataset/cnr_c60/saved-map/", save_path = "./ply_files/", file_name = "meme_2.h5"):
    if not os.path.exists(save_path):
        
        os.makedirs(save_path)

    
    # Load the pointcloud using GradSLAM
    full_path = load_path+file_name
    print("full_path ", full_path)
    pointclouds = Pointclouds.load_pointcloud_from_h5(load_path)

    # Convert the pointcloud to Open3D format
    # Assuming pointclouds has only one point cloud in the batch
    pcd = pointclouds.open3d(0)

    # Save the pointcloud as a .ply file using Open3D
    o3d.io.write_point_cloud(save_path+file_name[:-2]+"ply", pcd)

def get_pt_files(folder_path):
    # List all files in the given folder
    files = os.listdir(folder_path)
    
    # Filter files ending with .pt
    pt_files = [file for file in files if file.endswith('.pt')]
    
    return pt_files

def get_h5_files(folder_path):
    # List all files in the given folder
    files = os.listdir(folder_path)
    
    # Filter files ending with .pt
    h5_files = [file for file in files if file.endswith('.h5')]
    
    return h5_files


# Load the tensor from the .pt file
pt_folder = "./build_depth/dataset/cnr_c60/saved-feat/"
tensor_paths = get_pt_files(pt_folder)

load_path = "./build_depth/dataset/cnr_c60/saved-map/pointclouds/"
save_h5_to_ply(load_path=load_path, file_name = "pc_points.h5")


"""
for i in tqdm.tqdm(range(0,2)):
    load_path = "./build_depth/dataset/cnr_c60/query_"+str(i)+"/"
    save_h5_to_ply(load_path=load_path, file_name = "meme_"+str(i)+".h5")

h5_files = get_h5_files("./build_depth/dataset/cnr_c60/saved-map_old/pointclouds/")
for h5_file in h5_files:
    print("h5_file ", h5_file)
    save_h5_to_ply(file_name = h5_file)
"""

"""
for tensor_path in tensor_paths:
    print(tensor_path)
    tensor = torch.load(pt_folder+tensor_path)
    print(tensor.shape)
    input()
    # Ensure the tensor is on the CPU and convert to numpy
    tensor_np = tensor.cpu().numpy()

    # Assuming the tensor contains point cloud data
    # Check the shape and dimensions of the tensor to confirm it is suitable for a point cloud
    print(f"Tensor shape: {tensor_np.shape}")

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Assume the tensor contains only XYZ coordinates
    if tensor_np.shape[1] == 3:
        pcd.points = o3d.utility.Vector3dVector(tensor_np)

    # If the tensor contains XYZRGB or XYZRGBA, separate the points and colors
    elif tensor_np.shape[1] == 6 or tensor_np.shape[1] == 7:
        pcd.points = o3d.utility.Vector3dVector(tensor_np[:, :3])
        colors = tensor_np[:, 3:6]
        # Ensure colors are in the range [0, 1]
        if colors.max() > 1.0:
            colors /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the PointCloud to a PLY file
    ply_path = './ply_files/'+tensor_path[:-2]+"ply" 
    o3d.io.write_point_cloud(ply_path, pcd)

    print(f"Point cloud saved to {ply_path}")
"""

