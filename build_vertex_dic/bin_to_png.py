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

def read_file_to_tuples(file_path):
    result = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into components based on whitespace
            parts = line.split()
            
            # Convert the parts to the appropriate types (int for id, x, y)
            id_val = int(parts[0])
            x_val = int(parts[1])
            y_val = int(parts[2])
            
            # Append the tuple to the result list
            result.append((id_val, x_val, y_val))
    
    return result

def save_tuples_to_file(tuples, file_path):
    with open(file_path, 'w') as file:
        for id_val, colour in tuples:
            file.write(f"{id_val} {colour}\n")


# Function to process each bin file
def process_bin_file(bin_file_name, path_to_file= "./resources/dataset/cnr_c60/concat_feats/", save_dir= "./resources/dataset/cnr_c60/concat_feats/rgb_feats", image_height = 1080, image_width = 1920, force = False):
    print("saving: ",bin_file_name)
    bin_from_file = np.fromfile(path_to_file + bin_file_name, dtype=np.float32)
        
    if "_nz_feat.bin" in bin_file_name or force:
        image_height = int(len(bin_from_file)/1024)
        print(image_height)
        
    #np.savetxt('./resources/array_data.txt', y.reshape(1080, -1), delimiter=',')
    bin_file = torch.from_numpy(bin_from_file).reshape(image_height, image_width, -1)
    bin_file = bin_file.view(-1, bin_file.size()[2])

    print(bin_file.shape)

    pca = PCA(n_components=3)
    pca_r = pca.fit_transform(bin_file)
    pca_r = MinMaxScaler(feature_range=(0, 255), clip=True).fit_transform(pca_r)
    print(type(pca_r))
    
    if(bin_file_name=="all_feats.bin"):
        target_height, target_width = 1450, 2000
        # Create a list of the difference in dimensions
        pad_width = (1450*2000) #- len(pca_r)
        # Pad the array using np.pad
        #padded_image = np.pad(pca_r, pad_width, mode='constant', constant_values=0.0)
        padded_image = np.zeros((pad_width,3), dtype=np.float32)
        padded_image[:len(pca_r)] = pca_r[:len(pca_r)]
        print("len(padded_image) ", len(padded_image))
        padded_image = padded_image.reshape((target_height, target_width, 3))

        cv2.imwrite((save_dir + "/3_" + bin_file_name[:-3] + "png"), padded_image)
        
        """
        tolerance = 1e-5
        vid = 1655584
        txt_to_compare = np.loadtxt(path_to_file +str(vid)+"_vtx.txt", dtype=np.float32)
        all_feats = bin_from_file.reshape(int(len(bin_from_file)/1024) ,1024)
        arrays_equal = np.allclose(all_feats[vid], txt_to_compare, atol=tolerance)
        if(arrays_equal): print("all_feats[vertex_id] == txt_to_compare")
        else: print("all_feats[vertex_id] == txt_to_compare")
        """
        c=0
        result = []
        vertexs = read_file_to_tuples("./resources/dataset/cnr_c60/concat_feats/133468485245652555_vtx_coords.txt")
        pca_r = pca_r.reshape((-1,3))
        img_to_save = np.zeros((1080, 1920, 3), dtype=np.float32)
        
        for vid, x, y in vertexs:
            result.append((vid, pca_r[c, :3]))
            c+=1
            img_to_save[y,x,:] = pca_r[c, :3]

        cv2.imwrite((save_dir + "/" + bin_file_name[:-3] + "png"), img_to_save)
        save_tuples_to_file(result, "./resources/dataset/cnr_c60/concat_feats/133468485245652555_allfeat_colours.txt")
        print("\n")

    elif "_nz_feat.bin" in bin_file_name or force:
        img_to_save = np.zeros((1080, 1920, 3), dtype=np.float32)
        vertexs = read_file_to_tuples("./resources/dataset/cnr_c60/concat_feats/133468485245652555_vtx_coords.txt")
        
        to_compare = bin_from_file.reshape(int(len(bin_from_file)/1024), 1024)
        
        c = 0
        if force:
            pca_r = pca_r.reshape((1920, 1080, 3))
        
        result = []
        for vid, x, y in tqdm.tqdm(vertexs):
            if force:
                result.append((vid, pca_r[x, y, :3]))
                img_to_save[y,x,:] = pca_r[y, x :3]

            else:
                result.append((vid, pca_r[c, :3]))

                img_to_save[y,x,:] = pca_r[c, :3]
                txt_to_compare = np.loadtxt(path_to_file +"single_vertex/"+str(vid)+"_vtx.txt", dtype=np.float32)
                
                """
                arrays_equal = np.allclose(to_compare[c], txt_to_compare, atol=1e-5)

                if arrays_equal != True: 
                    print(vid)
                    print("to_compare[c] != txt_to_compare")
                #else: print("to_compare[c] == txt_to_compare")
                """
            c+=1

        print("len(result) ", len(result))
        save_tuples_to_file(result, "./resources/dataset/cnr_c60/concat_feats/133468485245652555_nz_colours.txt")
        
        cv2.imwrite((save_dir + "/" + bin_file_name[:-3] + "png"), img_to_save)

    else:
        pca_r = pca_r.reshape((image_height, image_width, 3))
        img_to_save = np.zeros((image_height, image_width, 3), dtype=np.float32)
        vertexs = read_file_to_tuples("./resources/dataset/cnr_c60/concat_feats/133468485245652555_vtx_coords.txt")
        result = []

        for vid, x, y in tqdm.tqdm(vertexs):
            img_to_save[y,x,:] = pca_r[y, x, :3]
            result.append((vid, pca_r[y, x, :3]))

        save_tuples_to_file(result, "./resources/dataset/cnr_c60/concat_feats/133468485245652555_wz_colours.txt")
        cv2.imwrite((save_dir + "/" + bin_file_name[:-3] + "png"), img_to_save)



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

    bin_files_names = find_all_extention_files(path_to_file, save_dir, extension='.bin')

    for file_name in tqdm.tqdm(bin_files_names, desc=" Coloring images based on features"):
        print(file_name)
        if file_name == "all_feats.bin":
            #print("skip")
            process_bin_file(file_name, path_to_file, save_dir, 2804853, 1)
        elif "_nz_feat.bin" in file_name:
            #process_bin_file("133468485245652555_0_1.bin", "./resources/dataset/cnr_c60/concat_feats/", save_dir, 1, 1, force = True)
            #process_bin_file("133468485245652555.bin", "./resources/dataset/cnr_c60/saved-feat/", save_dir, 1, 1, force = True)

            print("skip")
            #process_bin_file(file_name, path_to_file, save_dir, 1, 1)
        elif "_vtx.bin" not in file_name:
            print("skip")
            #process_bin_file(file_name, path_to_file, save_dir, image_height, image_width)

    """
    print("num_thread ", num_thread)
    with ProcessPoolExecutor(max_workers=num_thread) as executor:
        futures = [executor.submit(process_bin_file, (file_name)) for file_name in bin_files_names]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Coloring images based on features"):
            future.result()
    """
#convert_bin_to_img(path_to_file = "./resources/dataset/cnr_c60/saved-feat/", save_dir = "./resources/dataset/cnr_c60/open_clip_rgb")
convert_bin_to_img()

vertexs = read_file_to_tuples("./resources/dataset/cnr_c60/concat_feats/133468485245652555_vtx_coords.txt")

#1655584 1919 1076 
#1655623 1918 1078 

"""
for vid, x, y in tqdm.tqdm(vertexs):
    if(vid in [1655623, 1655584]):
        print(vid)
        txt_to_compare = np.loadtxt("./resources/dataset/cnr_c60/concat_feats/"+"single_vertex/"+str(vid)+"_vtx.txt", dtype=np.float32)
        print(txt_to_compare)
        #breakpoint()
"""

def compare_bin_txt(vertex_id, path_to_bin, path_to_txt, path_to_all_feat_bin, path_to_ts_bin, x_proj, y_proj):
    bin = np.fromfile(path_to_bin, dtype=np.float32)
    print ("len(bin) ", len(bin))
    txt = np.loadtxt(path_to_txt, dtype=np.float32) 
    print ("len(txt) ", len(txt))
    
    print("bin[0] ", bin[0], " txt[0] ",txt[0])
    # Set tolerance for checking up to the first 5 decimal places
    tolerance = 1e-5

    print("")
    # Check if arrays are equal up to 5 decimal places
    arrays_equal = np.allclose(bin, txt, atol=tolerance)

    if(arrays_equal): print("bin == txt")
    else: print("bin == txt")

    print("")
    all_feats = np.fromfile(path_to_all_feat_bin, dtype=np.float32)
    print ("len(all_feats) ", len(all_feats))
    print("len(all_feats)/1024 ",len(all_feats)/1024)
    all_feats = all_feats.reshape(int(len(all_feats)/1024) ,1024)
    arrays_equal = np.allclose(all_feats[vertex_id], bin, atol=tolerance)

    if(arrays_equal): print("all_feats[vertex_id] == bin")
    else: print("all_feats[vertex_id] == bin")

    print("")
    ts_feat = np.fromfile(path_to_ts_bin, dtype=np.float32)
    print ("len(ts_feat) ", len(ts_feat))
    print ("len(ts_feat)/1024 ", len(ts_feat)/1024)
    print ("len(ts_feat)/(1920*1080) ", len(ts_feat)/(1920*1080))
    ts_feat = ts_feat.reshape((1920, 1080, 1024))
    print("ts_feat.shape ", ts_feat.shape)
    arrays_equal = np.allclose(ts_feat[x_proj, y_proj, :], bin, atol=tolerance)
    if(arrays_equal): print("ts_feat[x_proj, y_proj, :] == bin")
    else: print("ts_feat[x_proj, y_proj, :] == bin")


vertex_id=1655584
path_to_bin="./resources/dataset/cnr_c60/concat_feats/"+str(vertex_id)+"_vtx.bin"
path_to_txt="./resources/dataset/cnr_c60/concat_feats/"+str(vertex_id)+"_vtx.txt"
path_to_all_feat_bin="./resources/dataset/cnr_c60/concat_feats/all_feats.bin"
timestamp = 133468485245652555  
x_proj, y_proj = 1919, 1076 
path_to_ts_bin="./resources/dataset/cnr_c60/concat_feats/"+str(timestamp)+"_0_1.bin"
#compare_bin_txt(vertex_id, path_to_bin, path_to_txt, path_to_all_feat_bin, path_to_ts_bin, x_proj, y_proj)

vertex_id=1655623
path_to_bin="./resources/dataset/cnr_c60/concat_feats/"+str(vertex_id)+"_vtx.bin"
path_to_txt="./resources/dataset/cnr_c60/concat_feats/"+str(vertex_id)+"_vtx.txt"
path_to_all_feat_bin="./resources/dataset/cnr_c60/concat_feats/all_feats.bin"
timestamp = 133468485245652555  
x_proj, y_proj = 1918, 1078
path_to_ts_bin="./resources/dataset/cnr_c60/concat_feats/"+str(timestamp)+"_0_1.bin"
#compare_bin_txt(vertex_id, path_to_bin, path_to_txt, path_to_all_feat_bin, path_to_ts_bin, x_proj, y_proj)

vertex_id=1655623
path_to_bin="./resources/dataset/cnr_c60/concat_feats/"+str(vertex_id)+"_vtx.bin"
path_to_txt="./resources/dataset/cnr_c60/concat_feats/"+str(vertex_id)+"_vtx.txt"
path_to_all_feat_bin="./resources/dataset/cnr_c60/concat_feats/all_feats.bin"
timestamp = 133468485245652555  
x_proj, y_proj = 1918, 1078
path_to_ts_bin="./resources/dataset/cnr_c60/concat_feats/"+str(timestamp)+"_0_1.bin"
#compare_bin_txt(vertex_id, path_to_bin, path_to_txt, path_to_all_feat_bin, path_to_ts_bin, x_proj, y_proj)

vertex_id=1657736
path_to_bin="./resources/dataset/cnr_c60/concat_feats/"+str(vertex_id)+"_vtx.bin"
path_to_txt="./resources/dataset/cnr_c60/concat_feats/"+str(vertex_id)+"_vtx.txt"
path_to_all_feat_bin="./resources/dataset/cnr_c60/concat_feats/all_feats.bin"
timestamp = 133468485245652555  
x_proj, y_proj = 1918, 1062
path_to_ts_bin="./resources/dataset/cnr_c60/concat_feats/"+str(timestamp)+"_0_1.bin"
#compare_bin_txt(vertex_id, path_to_bin, path_to_txt, path_to_all_feat_bin, path_to_ts_bin, x_proj, y_proj)


"""

import os
import torch
import tqdm

from pathlib import Path
import cv2
import numpy as np

def load_extrinsics(extrinsics_path):
    assert Path(extrinsics_path).exists()
    mtx = np.loadtxt(str(extrinsics_path), delimiter=',').reshape((4, 4))
    return mtx


def match_timestamp(target, all_timestamps):
    return np.argmin([abs(x - target) for x in all_timestamps])

def cam2world(points, rig2cam, rig2world):
    homog_points = np.hstack((points, np.ones((points.shape[0], 1))))
    cam2world_transform = rig2world @ np.linalg.inv(rig2cam)
    world_points = cam2world_transform @ homog_points.T
    return world_points.T[:, :3], cam2world_transform

def extract_timestamp(path):
    return int(path.split('.')[0])

def get_points_in_cam_space(path_to_xyz):
    #GET THE POINTS FROM THE XYZ FILE
    with open(path_to_xyz, 'r') as file:
        lines = file.readlines()

    # Assuming the format of each line in the file is: x y z (space-separated values)
    points = []
    for line in lines:
        # Split each line by spaces and convert to float
        values = list(map(float, line.strip().split()))
        points.append(values)
    
    # Convert list of points to a numpy array
    points_array = np.array(points)
    
    return points_array

def read_pv(path_to_pv):
    with open(path_to_pv, 'r') as file:
        lines = file.readlines()

    # First line: ox, oy, img_height, img_width
    ox, oy, img_width, img_height = map(float, lines[0].strip().split(','))
    print(f"ox: {ox}, oy: {oy}, img_height: {img_height}, img_width: {img_width}")

    # Create a dictionary to store the data with timestamp as key
    data_dict = {}
    
    # The rest: timestamp, fx, fy, extrinsics
    for line in lines[1:]:
        parts = line.strip().split(',')
        # First three values: timestamp, fx, fy
        timestamp = int(parts[0])
        fx = float(parts[1])
        fy = float(parts[2])
        # The rest 4x4 matrix (extrinsics)
        extrinsics = np.array(list(map(float, parts[3:]))).reshape(4, 4)

        # Store each entry in the dictionary with timestamp as key
        data_dict[timestamp] = (fx, fy, extrinsics)

    # Return the ox, oy, img_height, img_width, and the dictionary
    return ox, oy, img_height, img_width, data_dict

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Assuming the format of each line in the file is: x y z (space-separated values)
    points = []
    for line in lines:
        # Split each line by spaces and convert to float
        values = list(map(float, line.strip().split()))
        points.append(values)
    
    # Convert list of points to a numpy array
    points_array = np.array(points)
    
    return points_array

def save_ply(points, rgb, filename="output.ply"):
    # Ensure points and rgb have the same length
    assert points.shape[0] == rgb.shape[0], "Points and RGB arrays must have the same number of elements"
    
    # Concatenate points and RGB values
    ply_data = np.hstack([points, rgb])
    
    # Get the number of vertices (points)
    num_vertices = ply_data.shape[0]

    # Open the file to write
    with open(filename, 'w') as f:
        # Write the PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write the point and color data
        for row in ply_data:
            f.write(f"{row[0]} {row[1]} {row[2]} {int(row[3])} {int(row[4])} {int(row[5])}\n")


def save_single_pcloud(path,
                       fixed_extrinsics_path,
                       rig2world_transforms,
                       timestamp,
                       path_to_xyz
                       ):
    #rig2campath=fixed_extrinsics_path
    # from camera to rig space transformation (fixed)
    #rig2cam = load_extrinsics(rig2campath)
    #print("loaded fixed extrinsic")

    # Get xyz points in camera space
    points = get_points_in_cam_space(path_to_xyz)
    print(type(points))
    print("1: points[0] ", points[0])

    ox, oy, img_height, img_width, data = rig2world_transforms

    
    if timestamp in data:
        # if we have the transform from rig to world for this frame,
        # then put the point clouds in world space
        fx, fy, img_extrinsics = data[timestamp]

        rig2world = img_extrinsics
        print("extrinsics:\n", img_extrinsics)
        #print("fixed_extrinsics:\n", rig2cam)
        # print('Transform found for timestamp %s' % timestamp)
        #xyz, cam2world_transform = cam2world(points, rig2cam, rig2world)
        xyz = points
        rgb = None
    
        #target_id = match_timestamp(timestamp, pv_timestamps)
        #pv_ts = pv_timestamps[target_id]
        rgb_path = "./resources/dataset/cnr_c60/open_clip_rgb/"+str(timestamp)+".png"
        print(rgb_path)
        assert Path(rgb_path).exists()
        pv_img = cv2.imread(rgb_path)
        
        principal_point = [ox, oy]
        focal_lengths = [fx, fy]
        print("principal_point ", principal_point)
        print("focal_lengths ",focal_lengths)
        # Project from depth to pv going via world space
        rgb, depth = project_on_pv(xyz, pv_img, img_extrinsics, focal_lengths, principal_point)

        depth = (depth * 5000).astype(np.uint16)
        cv2.imwrite(("./resources/dataset/"+str(timestamp)+ "_depth.png"), (depth).astype(np.uint16))

        print("---")

        save_ply(points,rgb,"./resources/dataset/room_1st_coloured.ply")
        print("ply saved")
    else:
        print(f"Key '{timestamp}' does not exist in the dictionary")
            
def project_on_pv(points, pv_img, pv2world_transform, focal_length, principal_point):
    height, width, _ = pv_img.shape
    print("2: points[0] ", points[0])

    homog_points = np.hstack((points, np.ones(len(points)).reshape((-1, 1))))
    print("3: homog_points[0] ", homog_points[0], " shape ", homog_points[0].shape)
    print("extrinsics \n", pv2world_transform)
    world2pv_transform = np.linalg.inv(pv2world_transform)
    print("extrinsicInverse \n", world2pv_transform)
    points_pv = (world2pv_transform @ homog_points.T).T[:, :3]
    
    print("4: points_pv[0] ", points_pv[0], " shape ", points_pv[0].shape)


    intrinsic_matrix = np.array([[focal_length[0], 0, width-principal_point[0]], [
        0, focal_length[1], principal_point[1]], [0, 0, 1]])
    print("intrinsic:\n", intrinsic_matrix)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    xy, _ = cv2.projectPoints(points_pv, rvec, tvec, intrinsic_matrix, None)
    print("5: xy[0] ", xy[0])

    xy = np.squeeze(xy)
    print("6: xy[0] ", xy[0])

    xy[:, 0] = width - xy[:, 0]
    print("7: xy[0] ", xy[0])

    xy = np.floor(xy).astype(int)
    print("8: xy[0] ", xy[0])


    rgb = np.zeros_like(points)
    width_check = np.logical_and(0 <= xy[:, 0], xy[:, 0] < width)
    height_check = np.logical_and(0 <= xy[:, 1], xy[:, 1] < height)
    z_check = points_pv[:, 2] <= 0

    valid_ids = np.where(np.logical_and(np.logical_and(width_check, height_check), z_check))[0]
    print("valid_ids ", len(valid_ids))
    print(valid_ids)

    output_file = "./resources/dataset/cnr_c60/id_oracle_"+str(timestamp)+".txt"
    np.savetxt(output_file, valid_ids, fmt="%d")


    z = points_pv[valid_ids, 2]

    print("z\n",z)
    xy = xy[valid_ids, :]
    
    print("xy ", xy)

    depth_image = np.zeros((height, width))
    for i, p in enumerate(xy):
        depth_image[p[1], p[0]] = z[i]

    colors = pv_img[xy[:, 1], xy[:, 0], :]
    rgb[valid_ids, :] = colors[:, ::-1]# / 255.
    return rgb, depth_image

print("starting...")

path_to_image = "./resources/dataset/cnr_c60/open_clip_rgb"
fixed_extrinsics_path="./resources/dataset/dump_cnr_c60/Depth Long Throw_extrinsics.txt"

pv_path = "./resources/dataset/dump_cnr_c60/2023-12-12-105500_pv.txt"

if os.path.exists("./resources/dataset/cnr_c60"):
    print("Path exists")
else:
    print("Path does not exist")

extrinsics = read_pv(pv_path)

timestamp = 133468485245652555 
#timestamp = 133468485003417754
path_to_xyz = "./resources/dataset/room_1st.xyz"

save_single_pcloud(path=path_to_image, rig2world_transforms=extrinsics, timestamp=timestamp, path_to_xyz=path_to_xyz, fixed_extrinsics_path=fixed_extrinsics_path)
"""
"""

docker image build -t Paolo.Fasano/tesi_image:cpp_vertex_dic . && docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp_vertex_dic python3 ./build_vertex_dic/hl2_test.py

1: points[0]  [ 2.040295  1.37835  -3.836728]
2: points[0]  [ 2.040295  1.37835  -3.836728]
3: homog_points[0]  [ 2.040295  1.37835  -3.836728  1.      ]  shape  (4,)
extrinsics 
 [[-0.705878   -0.257086    0.660035    0.784161  ]
 [ 0.00707405  0.929208    0.369496    0.0440752 ]
 [-0.7083      0.265488   -0.654088    0.169949  ]
 [ 0.          0.          0.          1.        ]]
extrinsicInverse  
    [[-0.70587636  0.00707446 -0.70829784  0.67358341]
    [-0.25708544  0.9292036   0.26548664  0.11552235]
    [ 0.66003218  0.36949396 -0.65408496 -0.42269593]
    [ 0.          0.          0.          1.        ]]
4: points_pv[0]  [ 1.96068465 -0.14684003  3.94280249]  shape  (3,)
intrinsic:
 [[1449.09 0.00000e+00 973.106]
 [0.00000e+00 1450.89 510.933]
 [0.00000e+00 0.00000e+00 1.00000e+00]]
5: xy[0]  [[1693.71235156  456.89815242]]
6: xy[0]  [1693.71235156  456.89815242]
7: xy[0]  [226.28764844 456.89815242]
8: xy[0]  [226 456]

c++
1: vertex[0] x: 2.04029 y: 1.37835 z: -3.83673
2: vertexHomogeneous  2.04029
 1.37835
-3.83673
       1
extrinsic 
 -0.705878  -0.257086   0.660035   0.784161
0.00707405   0.929208   0.369496  0.0440752
   -0.7083   0.265488  -0.654088   0.169949
         0          0          0          1
extrinsicInverse 
 -0.705876 0.00707446  -0.708298   0.673583
 -0.257085   0.929204   0.265487   0.115522
  0.660032   0.369494  -0.654085  -0.422696
         0         -0          0          1
3: camCoords  1.96068
-0.14684
  3.9428
       1
intrinsic 1449.09       0 946.894
      0 1450.89 510.933
      0       0       1
dif_intrinsic 1449.09       0 973.106
      0 1450.89 510.933
      0       0       1
4: imageCoords 6677.97
1801.46
 3.9428
5: depthImage.cols - 1 - x,y [225 457]
"""


"""
//con zero
1655584 [ 0.76874941 -0.06947297 -0.02706397] 
1655623 [ 0.69608213 -0.14761794 -0.02927369] 
1657736 [ 0.01352059  0.11525474 -0.0571435 ] 
//solo features
1655584 [-0.03225992  0.0069279  -0.05134635] 
1655623 [-0.04836684  0.01440963 -0.00064754] 
1657736 [-0.06598222  0.02700539  0.04584193] 
//all features
1655584 [-0.22275302  0.03173727  0.17190883]
1655623 [0.02648076 0.36760471 0.17245532]
1657736 [-0.22275302  0.03173727  0.17190883]
"""