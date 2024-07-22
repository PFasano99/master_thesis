#include "build_vertex_dic.h"

int main(int argc, char* argv[])
{   
    string mesh_path = "./resources/dataset/room_1st.off";
    string dataset_path = "./resources/dataset/";
    string path_to_pv = "dump_cnr_c60/";
    string path_to_depth_folder = "cnr_c60/depth";
    string json_save_path = "cnr_c60/vertex_images_json";
    string ply_save_path = "cnr_c60/ply_files";
    string clip_feat_path = "cnr_c60/saved-feat";
    string bin_path = "cnr_c60/concat_feats";
    int n_threads = 64; 

    cout << "starting vertex_dic_build" << endl;
    Project_vertex_to_image projector = Project_vertex_to_image(mesh_path, dataset_path, n_threads);
    auto dict = projector.get_vetex_to_pixel_dict(path_to_pv, path_to_depth_folder, clip_feat_path, json_save_path, bin_path);
    //auto map = dict[133468485003417754];
    //HandleMesh mesh_handle = HandleMesh(mesh_path, 1, false);
    //mesh_handle.select_vertex_from_map(map, dataset_path+ply_save_path, "133468485003417754.ply"); //133468485386595138
}