#include "build_vertex_dic.h"
#include "build_vertex_unit_tests.h"

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
    string clip_feat_path = "cnr_c60/dinov2-saved-feat";
    string bin_path = "cnr_c60/concat_feats";
    string path_to_features=dataset_path+clip_feat_path;
    int n_threads = 128; 

    create_directory_if_not_exists(dataset_path + ply_save_path);
    create_directory_if_not_exists(dataset_path + bin_path);
    create_directory_if_not_exists(dataset_path + bin_path + "/rgb_feats");
    create_directory_if_not_exists(dataset_path + bin_path + "/single_vertex");

    Project_vertex_to_image projector = Project_vertex_to_image(mesh_path, dataset_path, n_threads);

    bool run_unit_test = true;
    if (run_unit_test)
    {
        cout << "starting vertex_dic_build unit test" << endl;
        projector.color_mesh_by_features(dataset_path+bin_path+"/rgb_feats/3_dinoV2_all_feats.png", dataset_path+bin_path+"/all_feats_fp_"+projector.getTimestamp()+".ply");
        //projector.color_mesh_by_features(dataset_path+bin_path+"/rgb_feats/3_all_feats.png", dataset_path+bin_path+"/all_feats_fp_"+projector.getTimestamp()+".ply");
        
        //uncomment the following code to perform the unit tests
        /*
        Unit_Test_BV unit_test = Unit_Test_BV(projector);
        //bool test_allFeats_ogData(string path_to_allFeats, string path_to_og, string path_to_coords, bool expected_result, string message = "Testing original features agianst the all_feats.bin features"){

        //unit_test.test_allFeats_ogData(dataset_path+bin_path+"/all_feats.bin", dataset_path+clip_feat_path+"/133468485245652555.bin", dataset_path+bin_path+"/133468485245652555_vtx_coords.txt", true);
        //unit_test.test_allFeats_ogData(dataset_path+bin_path+"/all_feats.bin", dataset_path+bin_path+"/133468485245652555_0_1.bin", dataset_path+bin_path+"/133468485245652555_vtx_coords.txt", true, "all_feats vs 0_1");
        unit_test.print_result();
        
        
        int dim1 = 19;//1920; 
        int dim2 = 10;//1080;
        int dim3 = 10;//1024;
        Eigen::Tensor<float, 3> tensor3D(dim1,dim2,dim3);
        unit_test.initialize_eigen_tensor(tensor3D);
        unit_test.test_bin_load(tensor3D, dataset_path+bin_path, true);
        unit_test.test_get_values_from_coords(tensor3D, 0, 0, true);
        unit_test.test_get_values_from_coords(tensor3D, dim1/2, dim2/2, true);
        unit_test.test_get_values_from_coords(tensor3D, dim1-1, dim2-1, true);

        Eigen::Tensor<float, 1> tensor1D(dim3);
        unit_test.initialize_eigen_tensor(tensor1D);
        unit_test.test_bin_load(tensor1D, dataset_path+bin_path, true);

        //bool test_concatenate_features(string path_to_mesh, vector<string> timestamps, string path_to_pv, string path_to_depth_folder, string clip_feat_path, string json_save_path, bool normalize_tensor, string path_to_features, bool expected_result, string message = "Testing sum and normalization"){
        vector<string> timestamps; timestamps.push_back("133468485245652555");
        //unit_test.test_concatenate_features(mesh_path, timestamps, path_to_pv, path_to_depth_folder, clip_feat_path, "", false, path_to_features, true, "Testing single ts call");
        //unit_test.test_concatenate_features(mesh_path, timestamps, path_to_pv, path_to_depth_folder, clip_feat_path, "", true, path_to_features, true, "Testing single ts call and normalization");

        unit_test.test_save_tensor_ordered(mesh_path, timestamps, path_to_pv, path_to_depth_folder, clip_feat_path, "", false, path_to_features, dataset_path+clip_feat_path, true, "Testing ordered saving");
        unit_test.test_projected_point("./resources/dataset/cnr_c60/id_oracle_133468485245652555.txt",mesh_path, timestamps, path_to_pv, path_to_depth_folder, clip_feat_path, "", true, "Testing projected points validitiy");

        unit_test.print_result();
        */

    }
    else{
        cout << "starting vertex_dic_build" << endl;
        
        map<long long, map<int, vector<vcg::Point2f>>> dict;
        projector.get_vetex_to_pixel_dict(dict, path_to_pv, path_to_depth_folder, clip_feat_path, json_save_path);

        auto tensors = projector.make_tensors(dict, path_to_features, bin_path, true, false, true);
    }

    
}
