#include "test_feat.h"

void create_directory_if_not_exists(const std::string& path) {
    if (!filesystem::exists(path)) {
        if (filesystem::create_directories(path)) {
            std::cout << "Directory created successfully: " << path << std::endl;
        } else {
            std::cerr << "Error: Failed to create directory: " << path << std::endl;
        }
    }
}

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
    string path_to_cos_similarity = "cnr_c60/cosine_similarity";
    string prompts_bin = "cnr_c60/prompt_feat";
    int n_threads = 64; 

    create_directory_if_not_exists("resources/cnr_c60/ply_files");

    cout << "Starting test_feat" << endl;
    Test_feat demo_q = Test_feat(mesh_path, dataset_path, n_threads, path_to_cos_similarity, bin_path+"/single_vertex", true);
    cout << "   Coloring 3d map" <<endl;
    std::vector<filesystem::path> bin_paths;
    Project_vertex_to_image().findFilesWithExtension(bin_paths, dataset_path+prompts_bin, ".bin");
    
    for (const auto& file : bin_paths) {
        Eigen::Tensor<float, 1> feature = demo_q.tensors[91231];
        Project_vertex_to_image().load_tensor_from_binary(feature,file,true);
        cout<<"feature.size() "<<feature.size()<<endl;
        cout<<"feature.dimension(0) "<< feature.dimension(0) << endl;
        string mesh_filename = "cosine_similarity_"+demo_q.getTimestamp()+".ply";

        demo_q.color_map_by_features(feature, dataset_path+ply_save_path+"/",mesh_filename);
    }    
    cout << "Done test_feat"<<endl;
}