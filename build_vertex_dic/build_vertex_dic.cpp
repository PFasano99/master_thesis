#include "build_vertex_dic.h"

int main(int argc, char* argv[])
{   
    string mesh_path = "./resources/dataset/room_1st.off";
    string dataset_path = "./resources/dataset/";
    string path_to_pv = "dump_cnr_c60/";
    string path_to_depth_folder = "cnr_c60/depth";
    string json_save_path = "cnr_c60/vertex_images_json";
    string ply_save_path = "cnr_c60/ply_files";
    string bin_path = "cnr_c60/concat_feats";
    int n_threads = 16; 
    cout << "starting vertex_dic_build" << endl;


    Project_vertex_to_image projector = Project_vertex_to_image(mesh_path, dataset_path, n_threads);
    auto dict = projector.get_vetex_to_pixel_dict(path_to_pv, path_to_depth_folder, json_save_path);
    auto map = dict[133468485003417754];
        
    //auto map = projector.json_to_map(dataset_path+json_save_path, "133468485006083337");
    //projector.print_map(read_json);
    HandleMesh mesh_handle = HandleMesh(mesh_path, 1, false);
    mesh_handle.select_vertex_from_map(map, dataset_path+ply_save_path, "133468485003417754.ply"); //133468485386595138
        
    //mesh_handle.select_vertex_from_map(map, dataset_path+ply_save_path, "133468485006083337.ply");

}