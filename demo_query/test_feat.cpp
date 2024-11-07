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
    string json_save_path = "cnr_c60/similarity/";
    string ply_save_path = "cnr_c60/ply_files";
    string bin_path = "cnr_c60/concat_feats";
    //string path_to_similarity = "cnr_c60/similarity";
    bool combine_clip_dinov2 = true;
    bool run_both_normal_and_fusion = true;
    string feature_estractor = "openClip"; //["dinoV2","openClip"]
    string feat_path = "";
    string openClip_feat_path = "/clip_all_feats.bin";
    string dinoV2_feat_path = "/dinoV2_all_feats.bin";
    if(feature_estractor == "openClip")
        feat_path = bin_path+openClip_feat_path;
    else if(feature_estractor == "dinoV2")
        feat_path = bin_path+dinoV2_feat_path;

    string prompts_bin = "cnr_c60/prompt_feat";
    int n_threads = 64; 
    int key = 161338;
    bool run_on_key = false;
    vector<string> similarity_metric_list = {"euclidean", "cosine", "manhattan", "pearson", "spearman"};
    string similarity_metric = similarity_metric_list[1];
    bool run_all_metrics = false;

    create_directory_if_not_exists(dataset_path+ply_save_path);
    create_directory_if_not_exists(dataset_path+json_save_path);

    cout << "Starting test_feat" << endl;
    Test_feat demo_q = Test_feat(mesh_path, dataset_path, n_threads, json_save_path, feat_path, true);
    
    cout << "   Coloring 3d map" <<endl;
    if (run_on_key && combine_clip_dinov2 == false){
        Eigen::Tensor<float, 1> feature = demo_q.tensors[key];
        bool check_Zero = true;
        for(int v = 0; v < feature.size(); v++){
            if(feature(v) != 0)
            {
                check_Zero = false;
                break;   
            }
        }

        if(check_Zero){
            cerr << "The feature selcted is all zeros, cant't perform "<<similarity_metric<<" calculation"<<endl;
            exit(1);
        }

        if(run_all_metrics){
            for(int m = 0; m < similarity_metric_list.size(); m++){
                similarity_metric = similarity_metric_list[m];
                string mesh_filename = "prompt_key_"+to_string(key)+"_"+similarity_metric+"_similarity_"+feature_estractor+"_"+demo_q.current_timestamp+".ply";
                demo_q.reset_timestamp();
                demo_q.color_map_by_features(feature, similarity_metric, dataset_path+ply_save_path+"/",mesh_filename);    
            }
        }
        else{
            string mesh_filename = "prompt_key_"+to_string(key)+"_"+similarity_metric+"_similarity_"+feature_estractor+"_"+demo_q.current_timestamp+".ply";
            demo_q.color_map_by_features(feature, similarity_metric, dataset_path+ply_save_path+"/",mesh_filename);
        }
    }
    else if (combine_clip_dinov2 == false || run_both_normal_and_fusion == true){
        cout << "Performing "<<feature_estractor<<"..." << endl;
        std::vector<filesystem::path> bin_paths;
        Project_vertex_to_image().findFilesWithExtension(bin_paths, dataset_path+prompts_bin, ".bin");
        int prompt_count = 0;
        for (const auto& file : bin_paths) 
        {
            Eigen::Tensor<float, 1> feature = demo_q.tensors[key];
            Project_vertex_to_image().load_tensor_from_binary(feature, file.u8string());
            bool check_Zero = true;
            for(int v = 0; v < feature.size(); v++){
                if(feature(v) != 0)
                {
                    check_Zero = false;
                    break;   
                }
            }

            if(check_Zero){
                cerr << "The feature selcted is all zeros, cant't perform "<<similarity_metric<<" calculation"<<endl;
                exit(1);
            }

            string prompt_name = file.filename().string();
            prompt_name.resize(prompt_name.size() - 4);

            if(run_all_metrics){
                for(int m = 0; m < similarity_metric_list.size(); m++){
                    similarity_metric = similarity_metric_list[m];
                    string mesh_filename = prompt_name+"_"+similarity_metric+"_similarity_"+feature_estractor+"_"+demo_q.current_timestamp+".ply";
                    demo_q.reset_timestamp();
                    demo_q.color_map_by_features(feature, similarity_metric, dataset_path+ply_save_path+"/",mesh_filename);    
                }
            }
            else{
                string mesh_filename = prompt_name+"_"+similarity_metric+"_similarity_"+feature_estractor+"_"+demo_q.current_timestamp+".ply";
                demo_q.color_map_by_features(feature, similarity_metric, dataset_path+ply_save_path+"/",mesh_filename);
            }

            prompt_count++;
        }    
    }    

    if (combine_clip_dinov2 == true){
        cout << "Performing CLIP and DinoV2 fusion..." << endl;
        std::vector<filesystem::path> bin_paths;
        Project_vertex_to_image().findFilesWithExtension(bin_paths, dataset_path+prompts_bin, ".bin");
        
        vector<Eigen::Tensor<float, 1>> clip_feat; 
        Project_vertex_to_image().load_all_tensors_from_bin(clip_feat, dataset_path+bin_path+openClip_feat_path, 1024);

        vector<Eigen::Tensor<float, 1>> dinoV2_feat;
        Project_vertex_to_image().load_all_tensors_from_bin(dinoV2_feat, dataset_path+bin_path+dinoV2_feat_path, 1024);

        for (const auto& file : bin_paths) 
        {
            Eigen::Tensor<float, 1> feature = demo_q.tensors[key];
            Project_vertex_to_image().load_tensor_from_binary(feature, file.u8string());
            bool check_Zero = true;
            for(int v = 0; v < feature.size(); v++){
                if(feature(v) != 0)
                {
                    check_Zero = false;
                    break;   
                }
            }

            if(check_Zero){
                cerr << "The feature selcted is all zeros, cant't perform "<<similarity_metric<<" calculation"<<endl;
                exit(1);
            }

            string prompt_name = file.filename().string();
            prompt_name.resize(prompt_name.size() - 4);

            string mesh_filename = prompt_name+"_"+similarity_metric+"_similarity_CLIP_DinoV2_fusion_"+demo_q.current_timestamp+".ply";
            demo_q.combine_clip_dino(clip_feat, dinoV2_feat, feature, similarity_metric, dataset_path+ply_save_path+"/",mesh_filename);
            
        }

    }

    cout << "Done test_feat"<<endl;
}