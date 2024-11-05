#include "../build_vertex_dic/build_vertex_dic.h"

class Test_feat{

    public: 
        vector<Eigen::Tensor<float, 1>> tensors;
    
    Project_vertex_to_image projector;
    string path_to_mesh;
    string path_to_dataset;
    int n_threads = 64;
    string path_to_cosine_similairty;


    string path_to_pv, path_to_depth_folder, clip_feat_path, json_save_path, bin_path;

    string current_timestamp = "";

    public:
        Test_feat(string mesh_path, string dataset_path, int threads, string save_json_path = "", string path_to_feats="", bool load_features = true){
            path_to_mesh = mesh_path;
            path_to_dataset = dataset_path;
            n_threads = threads;
            json_save_path = save_json_path;
            projector = Project_vertex_to_image(mesh_path, dataset_path, n_threads);
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, 1);

            current_timestamp = getTimestamp();

            if (load_features){
                cout << "   Loading all features from bin"<<endl;
                projector.load_all_tensors_from_bin(tensors, path_to_dataset+path_to_feats, 1024);
            }
        }
    
    
    public:
        Test_feat(string mesh_path, string dataset_path, int threads, string save_json_path, string pv_path, string depth_folder_path, string path_to_clip_feat, string path_to_bin, string path_to_feats, bool load_features = true){
            path_to_mesh = mesh_path;
            path_to_dataset = dataset_path;
            n_threads = threads;

            projector = Project_vertex_to_image(mesh_path, dataset_path, n_threads);

            path_to_pv = pv_path;
            path_to_depth_folder = depth_folder_path;
            clip_feat_path = path_to_clip_feat;
            json_save_path = save_json_path;
            bin_path = path_to_bin;
            
            current_timestamp = getTimestamp();

            if (load_features){
                projector.load_all_tensors_from_bin(tensors, path_to_dataset+path_to_feats);
            }
            else {
                map<long long, map<int, vector<vcg::Point2f>>> dict;
                projector.get_vetex_to_pixel_dict(dict, path_to_pv, path_to_depth_folder, clip_feat_path, json_save_path, false);
                //tensors = projector.make_tensors(dict, clip_feat_path, bin_path, false, false, false);
            }
        }
    
    public:
        void color_map_by_features(int feature_key, string similarity_metric = "cosine", string path_to_save_mesh = "cnr_c60/ply_files/", string mesh_filename="_similarity.ply"){    
            Eigen::Tensor<float, 1> query_feature = tensors[feature_key];
            color_map_by_features(query_feature, similarity_metric, path_to_save_mesh, similarity_metric+mesh_filename);
        }


    public:
        void color_map_by_features(Eigen::Tensor<float, 1>& feature, string similarity_metric = "cosine" , string path_to_save_mesh = "resources/cnr_c60/ply_files/", string mesh_filename="_similarity.ply", string json_file_name="_similarity.json"){
            
            HandleMesh mesh_handle = HandleMesh(path_to_mesh, 1, false);
            map<int, double> similarity_map;
            cout << "similaity metric is: " << similarity_metric << endl;
            
            calculate_similaity_metric(similarity_metric, similarity_map, feature, true, path_to_dataset+json_save_path);
            // Find min and max values in cosine_similarity_map in one pass
            auto min_max = std::minmax_element(similarity_map.begin(), similarity_map.end(),
                                            [](const auto& a, const auto& b) { return a.second < b.second; });
            double min_value = min_max.first->second;
            double max_value = min_max.second->second;
            cout << "min_value " << min_value << " max_value " << max_value << endl;
            vcg::Color4b white(255, 255, 255, 0);
            vcg::Color4b yellow(255, 255, 0, 0);
            vcg::Color4b green(0, 255, 0, 0);
            vcg::Color4b red(255, 0, 0, 0);
            vcg::Color4b black(0, 0, 0, 0);
            vcg::Color4b blue(0, 0, 255, 0);
            //set all vertex to color based on lower value 
            for(int i = 0; i < mesh_handle.mesh.vert.size(); i++){
                mesh_handle.mesh.vert[i].C() = white;
            }

            float threshold = max_value-(max_value/9);
            cout << "max_value " << max_value << "\n min_value " << min_value << endl;
            cout << "threshold " << threshold << endl;

            for(auto it = similarity_map.cbegin(); it != similarity_map.cend(); ++it){
                int key = it->first;   
                double value = it->second; 
                //if(similarity_map[key] > threshold && similarity_map[key] < 1)
                {
                    // Normalize the cosine similarity value between 0 and 1
                    double normalized_value = (value - min_value) / (max_value - min_value);
                     // Apply alpha = 255 for full opacity
                    mesh_handle.mesh.vert[key].C() = jetColorMap(normalized_value);
                    //cout << "color at: "<<key<<" is "<< int(mesh_handle.mesh.vert[key].C()[0]) << " "<< int(mesh_handle.mesh.vert[key].C()[1]) << " " << int(mesh_handle.mesh.vert[key].C()[2]) << endl;
                    
                }          
                //else
                {
                //    mesh_handle.mesh.vert[key].C() = white;
                    //cout << "color at: "<<key<<" is "<< int(mesh_handle.mesh.vert[key].C()[0]) << " "<< int(mesh_handle.mesh.vert[key].C()[1]) << " " << int(mesh_handle.mesh.vert[key].C()[2]) << endl;
                    
                }     
            }

            //mesh_handle.mesh.vert[1655584].C() = blue;

            cout<<" Saving mesh at: "<<path_to_save_mesh<<endl;
            if (!filesystem::exists(path_to_save_mesh)) {
                // Create the directory
                if (filesystem::create_directory(path_to_save_mesh)) {
                    std::cout << "Directory created successfully: "<< path_to_save_mesh << std::endl;
                } else {
                    std::cerr << "Error: Failed to create directory: " << path_to_save_mesh << std::endl;
                }
            }

            int mask = vcg::tri::io::Mask::IOM_VERTCOORD;
            mask |= vcg::tri::io::Mask::IOM_VERTQUALITY;
            mask |= vcg::tri::io::Mask::IOM_VERTCOLOR;
            mesh_handle.save_mesh(path_to_save_mesh+"/"+mesh_filename, mask);
            cout << "saved mesh at: " << path_to_save_mesh<<" as: "<<mesh_filename << endl;
        }

    public: 
        void calculate_similaity_metric(string similarity_metric, map<int, double>& similarity_map,  Eigen::Tensor<float, 1>& query_feature, bool save_to_json=false, string save_path="./", bool l2_normalize_data = false){

            cout<<"Calculating "<<similarity_metric<<" distance.. "<<endl;
            float done_similarity = 0;
            cout<<" Calculated "<< (done_similarity/tensors.size())*100 << "% | "<<static_cast<int>(done_similarity)<<"/"<<tensors.size()<<"\r" << std::flush;             
            
            Eigen::Tensor<float, 1>& tensor_a = query_feature;

            if(l2_normalize_data)
                projector.l2_normalization(tensor_a);

            for(int i = 0; i < tensors.size(); i++){
                Eigen::Tensor<float, 1>& tensor_b = tensors[i];
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

                if(l2_normalize_data){
                    projector.l2_normalization(tensor_b);
                }

                if (similarity_metric == "cosine")
                    similarity_map[i] = cosine_similarity(tensor_a, tensor_b);
                else if(similarity_metric == "euclidean")
                    similarity_map[i] = euclidean_distance(tensor_a, tensor_b);
                else if(similarity_metric == "manhattan")
                    similarity_map[i] = manhattan_distance(tensor_a, tensor_b);
                else if(similarity_metric == "pearson")
                    similarity_map[i] = pearson_correlation(tensor_a, tensor_b);
                else if(similarity_metric == "spearman")
                    similarity_map[i] = spearman_correlation(tensor_a, tensor_b);
                else{
                    cerr<<"Check the similarity metric chosen, avaliable metrics are:\n euclidean\n cosine" << endl;
                }
                done_similarity++;

                if (static_cast<int>(done_similarity) % static_cast<int>(tensors.size()/12) == 0)
                    cout<<" Calculated " << (done_similarity/tensors.size())*100 << "% | "<<static_cast<int>(done_similarity)<<"/"<<tensors.size()<<"\r" << std::flush;             
            }
            cout<<" Calculated " << (done_similarity/tensors.size())*100 << "% | "<<done_similarity<<"/"<<tensors.size()<<endl;             

            if(save_to_json)
                projector.map_to_json(similarity_map, save_path, similarity_metric+"_similarity_"+current_timestamp);

        }

    public:
        double cosine_similarity(Eigen::Tensor<float, 1>& tensor_a, Eigen::Tensor<float, 1>& tensor_b, bool using_normalized_data = true){

            double score = 0.0;

            if(using_normalized_data){
                //cout << "inner product in cosine similarity" << endl;
                std::vector<float> tensor_a_array = tensor_to_vector(tensor_a);
                std::vector<float> tensor_b_array = tensor_to_vector(tensor_b);
                score = std::inner_product(tensor_a_array.begin(), tensor_a_array.end(), tensor_b_array.begin(), 0.0);
            }
            else
            {
                double dot_prod = 0.0;
                double magnitude_a = 0.0;
                double magnitude_b = 0.0;
                for(int i = 0; i < tensor_a.size(); i++){
                    dot_prod += (tensor_a(i) * tensor_b(i));
                    magnitude_a += (tensor_a(i) * tensor_a(i));
                    magnitude_b += (tensor_b(i) * tensor_b(i));
                }   
                if(magnitude_a != 0 && magnitude_b != 0)
                    score = dot_prod / (sqrt(magnitude_a) * sqrt(magnitude_b));
            }
            

            return score;
        }        
   
    public: 
        double euclidean_distance(Eigen::Tensor<float, 1>& tensor_a, Eigen::Tensor<float, 1>& tensor_b){
            //Definition: Measures the straight-line distance between two points in Euclidean space
            double score = 0.0;
            double sum = 0;
            
            for(int i = 0; i < tensor_a.size(); i++){
                sum += pow((tensor_a(i) - tensor_b(i)), 2);
            } 
                           
            score = sqrt(sum);
            return score;
        }

    public:
        double manhattan_distance(Eigen::Tensor<float, 1>& tensor_a, Eigen::Tensor<float, 1>& tensor_b){
            //Definition: it measures the sum of absolute differences between corresponding elements.
            double score = 0.0;

            for(int i = 0; i < tensor_a.size(); i++){
                score += abs(tensor_a(i) - tensor_b(i));
            }   

            return score;
        }        

    public:
        double pearson_correlation(Eigen::Tensor<float, 1>& tensor_a, Eigen::Tensor<float, 1>& tensor_b){
            //Definition: Measures the linear correlation between two variables.

            double score = 0.0;
            
            Eigen::Tensor<float, 0> sum_a = tensor_a.sum();
            Eigen::Tensor<float, 0> sum_b = tensor_b.sum();

            double mean_a = sum_a() / tensor_a.size();      
            double mean_b = sum_b() / tensor_b.size();      
            double numerator = 0;
            double denominator_a = 0;
            double denominator_b = 0;

            for(int i = 0; i < tensor_a.size(); i++){
                numerator += (tensor_a(i)-mean_a)*(tensor_b(i)-mean_b);
                denominator_a += pow((tensor_a(i)-mean_a),2);
                denominator_b += pow((tensor_b(i)-mean_b),2); 
            }   

              // Check if the denominator is zero to avoid division by zero
            if (denominator_a == 0.0 || denominator_b == 0.0) {
                return 0.0;  // Return 0 or handle the case differently
            }

            score = (numerator)/(sqrt(denominator_a)*sqrt(denominator_b));
            return score;
        }      

    public:
        double spearman_correlation(Eigen::Tensor<float, 1>& tensor_a, Eigen::Tensor<float, 1>& tensor_b){
            //Definition: Measures the strength and direction of association between two ranked variables
            int n = tensor_a.size();
            if (n == 0) {
                return 0.0;  // Handle empty tensors
            }

            // Rank tensors a and b
            std::vector<int> rank_a = get_ranks(tensor_a);
            std::vector<int> rank_b = get_ranks(tensor_b);

            // Compute the sum of squared rank differences
            double sum_of_squared_diffs = 0.0;
            for (int i = 0; i < n; ++i) {
                sum_of_squared_diffs += pow((rank_a[i] - rank_b[i]), 2);
            }

            // Apply the Spearman correlation formula
            double score = 1.0 - (6.0 * sum_of_squared_diffs) / (n * (pow(n,2) - 1));

            return score;
        }

    private:
        // Helper function to compute ranks for a tensor
        std::vector<int> get_ranks(const Eigen::Tensor<float, 1>& tensor) {
            int n = tensor.size();
            std::vector<std::pair<float, int>> values_and_indices(n);

            // Store values along with their original indices
            for (int i = 0; i < n; ++i) {
                values_and_indices[i] = std::make_pair(tensor(i), i);
            }

            // Sort by value to rank them
            std::sort(values_and_indices.begin(), values_and_indices.end());

            // Create a vector to hold ranks
            std::vector<int> ranks(n);
            for (int i = 0; i < n; ++i) {
                ranks[values_and_indices[i].second] = i + 1; // Rank starts from 1
            }

            return ranks;
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

    public:
        std::string getTimestamp() {
            // Get current time
            auto now = std::chrono::system_clock::now();
            
            // Convert to time_t to break it down to calendar time
            std::time_t now_time = std::chrono::system_clock::to_time_t(now);
            
            // Create a tm struct to hold the broken-down time
            std::tm* local_time = std::localtime(&now_time);
            
            // Create a stringstream to format the timestamp
            std::stringstream timestamp;
            timestamp << std::put_time(local_time, "%d_%m_%H_%M_%S");
            
            return timestamp.str();
        }

    public:
        void print_tensor(Eigen::Tensor<float, 1> tensor, bool vertical){
            for(int i = 0; i < tensor.dimension(0); i++)
            {
                if (vertical)
                    cout << tensor(i) << endl;
                else
                    cout <<tensor(i) << " "; 
            }
            cout<<endl;
        }

    public:
        void reset_timestamp(){
            current_timestamp = getTimestamp();
        }

    public:
        vcg::Color4b map_colour_between(double normalized_value, vcg::Color4b colour1, vcg::Color4b colour2){
            normalized_value = std::clamp(normalized_value, 0.0, 1.0); // Ensure it's between 0 and 1

            // The closer the value is to 1 the colour2 it is, the closer it is to 0 the colour1, | colour2 == ok | colour1 == not ok |
            int r = static_cast<int>((1.0 - normalized_value) * colour1[0] + normalized_value * colour2[0]);
            int g = static_cast<int>((1.0 - normalized_value) * colour1[1] + normalized_value * colour2[1]);
            int b = static_cast<int>((1.0 - normalized_value) * colour1[2] + normalized_value * colour2[2]);

            vcg::Color4b newColor(r, g, b, 255); 
            return newColor;
        }

    // Helper function to map normalized_value to jet color map
    public:
        vcg::Color4b jetColorMap(double normalized_value) {
            // Clamp normalized_value to [0, 1] to ensure it’s within range
            normalized_value = std::clamp(normalized_value, 0.0, 1.0);

            // Initialize RGB components
            int r = 0, g = 0, b = 0;

            // Apply the piecewise function for "jet" colormap
            if (normalized_value <= 0.25) {
                r = 0;
                g = static_cast<int>(4.0 * 255.0 * normalized_value);  // ramping up green
                b = 255;
            } else if (normalized_value <= 0.5) {
                r = 0;
                g = 255;
                b = static_cast<int>(255.0 * (1.0 - 4.0 * (normalized_value - 0.25)));  // ramping down blue
            } else if (normalized_value <= 0.75) {
                r = static_cast<int>(4.0 * 255.0 * (normalized_value - 0.5));  // ramping up red
                g = 255;
                b = 0;
            } else {
                r = 255;
                g = static_cast<int>(255.0 * (1.0 - 4.0 * (normalized_value - 0.75)));  // ramping down green
                b = 0;
            }

            // Return the final color with full opacity
            return vcg::Color4b(r, g, b, 255);
        }


};