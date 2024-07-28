#include "../build_vertex_dic/build_vertex_dic.h"

class Test_feat{

    std::map<int, Eigen::Tensor<float, 1>> tensors;
    Project_vertex_to_image projector;
    string path_to_mesh;
    string path_to_dataset;
    int n_threads;
    string path_to_cosine_similairty;


    string path_to_pv, path_to_depth_folder, clip_feat_path, json_save_path, bin_path;


    public:
        Test_feat(string mesh_path, string dataset_path, int threads, string cosine_similarity_path = "", string path_to_feats="", bool load_features = true){
            path_to_mesh = mesh_path;
            path_to_dataset = dataset_path;
            n_threads = threads;
            path_to_cosine_similairty = cosine_similarity_path;
            projector = Project_vertex_to_image(mesh_path, dataset_path, n_threads);

            if (load_features){
                tensors = projector.load_all_tensors_from_bin(path_to_dataset+path_to_feats);
            }
        }
    
    public:
        Test_feat(string mesh_path, string dataset_path, int threads, string cosine_similarity_path, string pv_path, string depth_folder_path, string path_to_clip_feat, string save_json_path, string path_to_bin, string path_to_feats, bool load_features = true){
            path_to_mesh = mesh_path;
            path_to_dataset = dataset_path;
            n_threads = threads;
            path_to_cosine_similairty = cosine_similarity_path;

            projector = Project_vertex_to_image(mesh_path, dataset_path, n_threads);

            path_to_pv = pv_path;
            path_to_depth_folder = depth_folder_path;
            clip_feat_path = path_to_clip_feat;
            json_save_path = save_json_path;
            bin_path = path_to_bin;

            if (load_features){
                tensors = projector.load_all_tensors_from_bin(path_to_dataset+path_to_feats);
            }
            else {
                auto dict = projector.get_vetex_to_pixel_dict(path_to_pv, path_to_depth_folder, clip_feat_path, json_save_path, false);
                tensors = projector.make_tensors(dict, clip_feat_path, bin_path, false);
            }
        }

    public:
        void color_map_by_features(int feature_key, string path_to_save_mesh = "cnr_c60/ply_files", string mesh_filename="/cosine_similarity.ply"){
            
            Eigen::Tensor<float, 1> query_feature = tensors[feature_key];

            color_map_by_features(query_feature, path_to_save_mesh, mesh_filename);

        }


    public:
        void color_map_by_features(Eigen::Tensor<float, 1>& feature, string path_to_save_mesh = "cnr_c60/ply_files", string mesh_filename="/cosine_similarity.ply"){

            HandleMesh mesh_handle = HandleMesh(path_to_mesh, 1, false);
            map<int, double> cosine_similarity_map = cosine_similarity(feature, true, path_to_dataset+path_to_cosine_similairty);
            
            double min_value = std::numeric_limits<double>::max();
            double max_value = std::numeric_limits<double>::lowest();
            
            for (const auto& pair : cosine_similarity_map) {
                if (pair.second < min_value) {
                    min_value = pair.second;
                }
                else if (pair.second > max_value) {
                    max_value = pair.second;
                }
            }

            vcg::Color4b white(255, 255, 255, 0);
            //set all vertex to color based on lower value 
            //for(int i = 0; i < mesh_handle.mesh.vert.size(); i++){
            //    mesh_handle.mesh.vert[i].C() = white;
            //}

            vector<vcg::Point3f> vertexes;
            float threshold = max_value-(max_value/3);
            cout << "threshold " << threshold << endl;
            for(auto it = cosine_similarity_map.cbegin(); it != cosine_similarity_map.cend(); ++it){
                int key = it->first;   
                //mesh_handle.mesh.vert[key].Q() = cosine_similarity_map[key];
                if(cosine_similarity_map[key] > threshold){
                    int opacity =  1 + static_cast<int>((cosine_similarity_map[key] - min_value) * 254.0 / (max_value - min_value));
                    //cout << opacity << endl;
                    //vcg::Color4b newColor(255, 0, 0, 255);
                    cout << "color at: "<<key<<" was "<< int(mesh_handle.mesh.vert[key].C()[0]) << endl;
                    vcg::Color4b newColor(opacity, 0, 0, 255);
                    mesh_handle.mesh.vert[key].C() = newColor;
                    cout << "color at: "<<key<<" is "<< int(mesh_handle.mesh.vert[key].C()[0]) << endl;
                    vertexes.push_back(mesh_handle.mesh.vert[key].P());

                }       
                else{
                    mesh_handle.mesh.vert[key].C() = white;
                }         
            }

            mesh_handle.visualize_points_in_mesh(vertexes[0], vertexes, path_to_dataset+path_to_save_mesh+"/cosine_similarity_vert.ply");

            cout<<" Saving mesh at: "<<path_to_dataset+path_to_save_mesh<<endl;
            if (!filesystem::exists(path_to_dataset+path_to_save_mesh)) {
                // Create the directory
                if (filesystem::create_directory(path_to_dataset+path_to_save_mesh)) {
                    std::cout << "Directory created successfully: "<< path_to_dataset+path_to_save_mesh << std::endl;
                } else {
                    std::cerr << "Error: Failed to create directory: " << path_to_dataset+path_to_save_mesh << std::endl;
                }
            }

            int mask = vcg::tri::io::Mask::IOM_VERTCOORD;
            mask |= vcg::tri::io::Mask::IOM_VERTQUALITY;
            mask |= vcg::tri::io::Mask::IOM_VERTCOLOR;
            mesh_handle.save_mesh(path_to_dataset+path_to_save_mesh+mesh_filename, mask);

        }



    public: 
        map<int, double> cosine_similarity(Eigen::Tensor<float, 1>& query_feature, bool save_to_json=false, string save_path="./"){

            map<int, double> cosine_similarity_map;
            cout<<"Calculating cosine similarity.. "<<endl;
            float done_cosine_similarity = 0;

            cout<<" Calculated "<<std::setprecision(3) << std::fixed<< done_cosine_similarity/tensors.size()*100 << "% | "<<static_cast<int>(done_cosine_similarity)<<"/"<<tensors.size()<<"\r" << std::flush;             
            for(auto it = tensors.cbegin(); it != tensors.cend(); ++it){
                int key = it->first;   
                
                cosine_similarity_map[key] = cosine_similarity(query_feature, tensors[key], false);
                done_cosine_similarity++;

                if (static_cast<int>(done_cosine_similarity) % static_cast<int>(tensors.size()/12) == 0)
                    cout<<" Calculated "<<std::setprecision(3) << std::fixed<< done_cosine_similarity/tensors.size()*100 << "% | "<<static_cast<int>(done_cosine_similarity)<<"/"<<tensors.size()<<"\r" << std::flush;             
            }
            cout<<" Calculated "<<std::setprecision(3) << 100 << "% | "<<tensors.size()<<"/"<<tensors.size()<<endl;             

            if(save_to_json)
                projector.map_to_json(cosine_similarity_map, save_path, "cosine_similarity");

            return cosine_similarity_map;
        }

    public:
        double cosine_similarity(Eigen::Tensor<float, 1>& tensor_a, Eigen::Tensor<float, 1>& tensor_b, bool using_normalized_data = true, bool l2_normalize_data = false){

            if (tensor_a.size() != tensor_b.size()){
                cout<<"Tensors sizes mismatch, smaller tensor will be padded"<<endl;

                int max_size = max(tensor_a.size(), tensor_b.size());

                if (tensor_a.size() < max_size){
                    for(int i = tensor_a.size(); i < max_size; i++){
                        tensor_a(i) = 0.0;
                    }
                }
                else{
                    for(int i = tensor_b.size(); i < max_size; i++){
                        tensor_b(i) = 0.0;
                    }
                }   
            }

            double score = 0.0;

            if(l2_normalize_data){
                projector.l2_normalization(tensor_a);
                projector.l2_normalization(tensor_a);
                using_normalized_data = true;
            }

            std::vector<float> tensor_a_array = tensor_to_vector(tensor_a);
            std::vector<float> tensor_b_array = tensor_to_vector(tensor_b);
            //since ||tensor_a|| = 1 and ||tensor_b|| = 1 due to the data being normalized the cosine similarity formula can be simplified ad the dot product of tensor_a (*) tensor_b 
            
            if(using_normalized_data){
                score = std::inner_product(tensor_a_array.begin(), tensor_a_array.end(), tensor_a_array.begin(), 0.0);
            }
            else{
                double dot_prod = 0.0;
                double magnitude_a = 0.0;
                double magnitude_b = 0.0;
                for(int i = 0; i < tensor_a_array.size(); i++){
                    dot_prod += (tensor_a_array[i] * tensor_b_array[i]);
                    magnitude_a += (tensor_a_array[i] * tensor_a_array[i]);
                    magnitude_b += (tensor_b_array[i] * tensor_b_array[i]);
                }   
                score = dot_prod / (sqrt(magnitude_a) * sqrt(magnitude_b));
            }


            return score;
        }

    public:
        std::vector<float> tensor_to_vector(const Eigen::Tensor<float, 3>& tensor) {
            std::vector<float> vec(tensor.size());
            std::copy(tensor.data(), tensor.data() + tensor.size(), vec.begin());
            return vec;
        }

    public:
        std::vector<float> tensor_to_vector(const Eigen::Tensor<float, 1>& tensor) {
            std::vector<float> vec(tensor.size());
            std::copy(tensor.data(), tensor.data() + tensor.size(), vec.begin());
            return vec;
        }

};