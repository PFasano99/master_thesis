#include "test_feat.h"


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
    int n_threads = 64; 

    cout << "Starting test_feat" << endl;
    Test_feat demo_q = Test_feat(mesh_path, dataset_path, n_threads, path_to_cos_similarity, bin_path+"/single_vertex", true);

    auto bin_paths = Project_vertex_to_image().findFilesWithExtension(dataset_path+"cnr_c60/prompt_feat", ".bin");
    for (const auto& file : bin_paths) {
        Eigen::Tensor<float, 1> feature = Project_vertex_to_image().load_tensor_from_binary(file,true);
        cout<<"feature.size() "<<feature.size()<<endl;
        cout<<"feature.dimension(0) "<< feature.dimension(0) << endl;
        demo_q.color_map_by_features(feature);
    }    
}