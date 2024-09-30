/*
    This class has been made to perform some unit testing on the Project_vertex_to_image class
*/
//#include "../build_depth/build_depth.h"
//#include "build_vertex_dic.h"
#include "random"
#include <iomanip>  // For std::setw

class Unit_Test_BV{

    Project_vertex_to_image projector = Project_vertex_to_image();
    bool log_results;
    public: int passed_tests = 0;
    public: vector<bool> test_results;
    public: vector<bool> expected_results;
    public: vector<string> test_result_message;

    public: Unit_Test_BV(Project_vertex_to_image projector_vti, bool log = true){
        projector = projector_vti;
        log_results = log;
    }

    public: void add_result(bool result, bool expected_result, string function_name = "", string message = ""){
        test_results.push_back(result);
        expected_results.push_back(expected_result);
        string tr_message = "The test ["+to_string(test_results.size())+"] "+ function_name +" was: ";
        if (result == expected_result){
            tr_message+="PASS | ";
            passed_tests++;
        }
        else
            tr_message+="FAIL | ";


        tr_message+="expected: "+to_string(expected_result)+" got:"+to_string(result)+" | "+message;

        test_result_message.push_back(tr_message);
    }

    public: void print_result(){
        if(log_results && test_results.size()>0){
            int column_w = 96;
            string separator = "";
            for(int s = 0; s < column_w; s++)
                separator+="-";

            float pass_percentage = (static_cast<float>(passed_tests) / test_results.size()) * 100.0f;
            cout << separator<<endl;
            cout << "Test run: " << test_results.size() << endl;
            cout << "   Passed: " << passed_tests << " | " << pass_percentage << "%" << endl;
            cout << "   Failed: " << test_results.size()-passed_tests << " | " << 100-pass_percentage << "%" << endl;
            cout << separator<<endl;
            cout << "LOG:"<<endl;

            for(int r = 0; r < test_results.size(); r++){
                cout << "   >" << test_result_message[r] << endl;
            }
            cout << separator<<endl;
            cout<<endl;
        }
        else{
            cout << "No results were logged, to log results be sure to set log to true and to run some tests." << endl;
        }
    }

    /*given a tensor, saves it and than load it back to check if any errors occur*/
    
    public: 
        bool test_bin_load(const Eigen::Tensor<float, 3>& tensor, string save_path, bool expected_result, string message = "Eigen::Tensor<float, 3>"){
            projector.save_tensor_to_binary(tensor, save_path, "unit_test_load.bin");
            string file_path = save_path+"/unit_test_load.bin";

            Eigen::Tensor<float,3> loaded_Tensor;
            projector.load_tensor_from_binary(loaded_Tensor, file_path, tensor.dimension(0), tensor.dimension(1), tensor.dimension(2));

            bool result = areTensorsEqual(tensor, loaded_Tensor);

            // Check if the file exists
            if (std::filesystem::exists(file_path)) {
                // Delete the file
                std::filesystem::remove(file_path);
            }

            if (log_results)
                add_result(result, expected_result, "test_bin_load", message);                 

            if (result == expected_result)
                return true;
            return false;
        } 
    
    public: 
        bool test_bin_load(const Eigen::Tensor<float, 1>& tensor, string save_path, bool expected_result, string message = "Eigen::Tensor<float, 1>"){
            projector.save_tensor_to_binary(tensor, save_path, "unit_test_load.bin");
            string file_path = save_path+"/unit_test_load.bin";
            
            Eigen::Tensor<float,1> loaded_Tensor;
            projector.load_tensor_from_binary(loaded_Tensor, file_path, tensor.dimension(0));

            bool result = areTensorsEqual(tensor, loaded_Tensor);

            // Check if the file exists
            if (std::filesystem::exists(file_path)) {
                // Delete the file
                std::filesystem::remove(file_path);
            }

            if (log_results)
                add_result(result, expected_result, "test_bin_load", message);                 

            if (result == expected_result)
                return true;
            return false;
        } 
        

    public:
        bool test_get_values_from_coords(const Eigen::Tensor<float, 3>& tensor, int x, int y, bool expected_result, string message = "Retrive Eigen::Tensor<float, 1> from Eigen::Tensor<float, 3>"){
            
            Eigen::Tensor<float, 1> result_tensor = projector.get_values_from_coordinates(tensor, x, y); 

            bool result = areTensorsEqual(result_tensor, tensor, x, y);

            if (log_results)
                add_result(result, expected_result, "test_get_values_from_coords", message);                 

            if (result == expected_result)
                return true;
            
            return false;
        }


    public:
        bool test_concatenate_features(string path_to_mesh, vector<string> timestamps, string path_to_pv, string path_to_depth_folder, string clip_feat_path, string json_save_path, bool normalize_tensor, string path_to_features, bool expected_result, string message = "Testing sum and normalization"){
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, 1, false);

            map<long long, map<int, vector<vcg::Point2f>>> map_vertex;
            projector.get_vetex_to_pixel_dict(map_vertex, path_to_pv, path_to_depth_folder, clip_feat_path, json_save_path, false);

            Eigen::Tensor<float, 1> values(1024); 
            values.setZero();
            std::vector<Eigen::Tensor<float, 1>> tensors(mesh_handler.mesh.vert.size(), values);
        
            //void concatenate_features(std::vector<Eigen::Tensor<float, 1>>& tensors, map<long long, map<int, vector<vcg::Point2f>>>& map_vertex, std::vector<string> timestamps, bool normalize_tensors = true, string path_to_features="./resources/dataset/cnr_c60/saved-feat", string path_to_json="./resources/dataset/cnr_c60/vertex_images_json"){
            projector.concatenate_features(tensors, map_vertex, timestamps, normalize_tensor);

            Eigen::Tensor<float, 3> tensor; 
            projector.load_tensor_from_binary(tensor, path_to_features+"/"+timestamps[0]+".bin");

            std::map<int, std::vector<Point2f>> vertex_image_json = map_vertex[stoll(timestamps[0])]; 
            bool result = false;
            for(auto it = vertex_image_json.cbegin(); it != vertex_image_json.cend(); ++it){
                int key = it->first;
                vcg::Point2f p2f_json = vertex_image_json[key][0];

                Eigen::Tensor<float,1> resulting_tensor = projector.get_values_from_coordinates(tensor, p2f_json[0], p2f_json[1]);
                if(normalize_tensor){
                   projector.l2_normalization(resulting_tensor);
                }

                result = areTensorsEqual(resulting_tensor, tensors[key]);
                if (result == false)
                    break;
            }
            
            if (log_results)
                add_result(result, expected_result, "test_concatenate_features", message);                 

            if (result == expected_result)
                return true;
            
            return false;
        }

    public:
        bool test_save_tensor_ordered(string path_to_mesh, vector<string> timestamps, string path_to_pv, string path_to_depth_folder, string clip_feat_path, string json_save_path, bool normalize_tensor, string path_to_features, string original_file_path, bool expected_result, string message = "Testing ordered saving"){
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, 1, false);

            map<long long, map<int, vector<vcg::Point2f>>> map_vertex;
            projector.get_vetex_to_pixel_dict(map_vertex, path_to_pv, path_to_depth_folder, clip_feat_path, json_save_path, false);

            Eigen::Tensor<float, 1> values(1024); 
            values.setZero();
            std::vector<Eigen::Tensor<float, 1>> tensors(mesh_handler.mesh.vert.size(), values);
        
            //void concatenate_features(std::vector<Eigen::Tensor<float, 1>>& tensors, map<long long, map<int, vector<vcg::Point2f>>>& map_vertex, std::vector<string> timestamps, bool normalize_tensors = true, string path_to_features="./resources/dataset/cnr_c60/saved-feat", string path_to_json="./resources/dataset/cnr_c60/vertex_images_json"){
            projector.concatenate_features(tensors, map_vertex, timestamps, normalize_tensor);
            
            //void save_tensor_ordered(std::vector<Eigen::Tensor<float, 1>>& tensors, map<long long, map<int, vector<vcg::Point2f>>>& vertex_map, long long timestamp, string save_path, int dim1 = 1080, int dim2 = 1920, int dim3 = 1024)
            projector.save_tensor_ordered(tensors, map_vertex, stoll(timestamps[0]), "./resources/dataset/cnr_c60/");

            string file_path = "./resources/dataset/cnr_c60/"+timestamps[0]+"_0_1.bin";

            Eigen::Tensor<float, 3> saved_tensor; 
            projector.load_tensor_from_binary(saved_tensor, file_path);

            // Check if the file exists
            if (std::filesystem::exists(file_path)) {
                // Delete the file
                std::filesystem::remove(file_path);
            }

            Eigen::Tensor<float, 3> original_tensor; 
            projector.load_tensor_from_binary(original_tensor, original_file_path+"/"+timestamps[0]+".bin");

            std::map<int, std::vector<Point2f>> vertex_image_json = map_vertex[stoll(timestamps[0])]; 
            bool result = false;
            for(auto it = vertex_image_json.cbegin(); it != vertex_image_json.cend(); ++it){
                int key = it->first;
                vcg::Point2f p2f_json = vertex_image_json[key][0];

                Eigen::Tensor<float,1> resulting_tensor_og = projector.get_values_from_coordinates(original_tensor, p2f_json[0], p2f_json[1]);
                Eigen::Tensor<float,1> resulting_tensor_svd = projector.get_values_from_coordinates(saved_tensor, p2f_json[0], p2f_json[1]);
                
                result = areTensorsEqual(resulting_tensor_og, resulting_tensor_svd);
                if (result == false)
                    break;
            }

            if (log_results)
                add_result(result, expected_result, "test_save_tensor_ordered", message);                 

            if (result == expected_result)
                return true;
            
            return false;            

        }


    public:
        bool areTensorsEqual(const Eigen::Tensor<float, 3>& tensor1, const Eigen::Tensor<float, 3>& tensor2, bool verbose = false) {
            // Step 1: Check if the dimensions match
             // Check if the dimensions are the same
            if (tensor1.dimension(0) != tensor2.dimension(0) ||
                tensor1.dimension(1) != tensor2.dimension(1) ||
                tensor1.dimension(2) != tensor2.dimension(2)) {
                    if (verbose)
                        std::cout << "Tensors dimension mismatch" << std::endl;
                return false;
            }

            // Check if the values are the same
            for (int i = 0; i < tensor1.dimension(0); ++i) {
                for (int j = 0; j < tensor1.dimension(1); ++j) {
                    for (int k = 0; k < tensor1.dimension(2); ++k) {
                        if (tensor1(i, j, k) != tensor2(i, j, k)) {
                            if (verbose)
                                std::cout << "Tensors mismatch at (" << i << ", " << j << ", " << k << ")" << std::endl;
                            return false;
                        }
                    }
                }
            }

            // If dimensions and values match, the tensors are equal
            if (verbose)
                std::cout << "Tensors match" << std::endl;
            return true;
        }
    
    public:
        bool areTensorsEqual(const Eigen::Tensor<float, 1>& tensor1, const Eigen::Tensor<float, 1>& tensor2) {
            // Check if the dimensions are the same
            if (tensor1.dimension(0) != tensor2.dimension(0)) {
                cout << "tensors dimension mismatch "<< endl;
                return false;
            }

            // Check if the values are the same
            for (int i = 0; i < tensor1.dimension(0); ++i) {
                if (tensor1(i) != tensor2(i)) {
                    cout << "tensors mismatch "<< endl;
                    return false;
                }
            }
            return true;
        }
    
    public:
        bool areTensorsEqual(const Eigen::Tensor<float, 1>& tensor1, const Eigen::Tensor<float, 3>& tensor2, int x, int y) {
            // Check if the dimensions are the same
            if (tensor1.dimension(0) != tensor2.dimension(2)) {
                cout << "tensors dimension mismatch "<< endl;
                return false;
            }

            // Check if the values are the same
            for (int i = 0; i < tensor1.dimension(0); ++i) {
                if (tensor1(i) != tensor2(x,y,i)) {
                    cout << "tensors mismatch at:"<< i << endl;
                    return false;
                }
            }

            // If dimensions and values match, the tensors are equal
            //cout << "tensors match "<< endl;
            return true;
        }
    

    /*Initialize a tensor with random numbers in range*/
    public:
        void initialize_eigen_tensor(Eigen::Tensor<float, 3>& tensor, float min = 0.0f, float max = 10.0f){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 gen(rd()); // seed the generator
            std::uniform_real_distribution<> distr(min, max); // define the range

            for(int h = 0; h < tensor.dimension(0); h++){
                for(int w = 0; w < tensor.dimension(1); w++){
                    for(int v = 0; v < tensor.dimension(2); v++){
                        tensor(h,w,v) = distr(gen);
                    } 
                }
            }
        }
    
    public:
        void initialize_eigen_tensor(Eigen::Tensor<float, 2>& tensor, float min = 0.0f, float max = 10.0f){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 gen(rd()); // seed the generator
            std::uniform_real_distribution<> distr(min, max); // define the range

            for(int h = 0; h < tensor.dimension(0); h++){
                for(int w = 0; w < tensor.dimension(1); w++){
                    tensor(h,w) = distr(gen);
                }
            }
        }
    
    public:
        void initialize_eigen_tensor(Eigen::Tensor<float, 1>& tensor, float min = 0, float max = 10){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 gen(rd()); // seed the generator
            std::uniform_real_distribution<> distr(min, max); // define the range

            for(int h = 0; h < tensor.dimension(0); h++){
                tensor(h) = distr(gen);
            }
        }
};