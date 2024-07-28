#include "../build_depth/build_depth.h"
#include <map>
#include <utility>      // std::pair, std::make_pair
#include <nlohmann/json.hpp>
#include <chrono>

using namespace std::chrono;
using json = nlohmann::json;
using namespace std;
using namespace vcg;

class Project_vertex_to_image
{

    private: int n_threads = 32;
    private: bool verbose = false;
    private: string path_to_mesh;
    private: string path_to_dataset;

    
    public:
        Project_vertex_to_image()
        {
            
        }
    
    public:
        Project_vertex_to_image(string mesh_path, string dataset_path, int threads = 4, bool isVerbose = false)
        {
            n_threads = threads;
            omp_set_num_threads(n_threads);
            verbose = isVerbose;
            path_to_mesh = mesh_path;
            path_to_dataset = dataset_path;
        }

    /*
        @param const std::string& filePath, path to the depth image to read

        @description
            given the path to a depth.png image, use openvc to read the image

        @return cv::Mat representing the image 
    */
    public:
        cv::Mat loadDepthImage(const std::string& filePath) {
            cv::Mat depthImage = cv::imread(filePath, cv::IMREAD_UNCHANGED);
            if (depthImage.empty()) {
                throw std::runtime_error("Could not load depth image at path: "+ filePath);
            }
            return depthImage;
        }

    /*
        All the following methods called add_value_to_map work the same way with sligltly different parameters 
        @param

        @description
            given a pointer to a map, a key and a value, adds the value in the map at the given key

        @return
    */
    public: 
        void add_value_to_map(map<int,vector<vcg::Point2f>>& map, int key, vector<vcg::Point2f> values){
            for(int i = 0; i < values.size(); i++){
                add_value_to_map(map, key, values[i]);
            }
        }

    public:
        void add_value_to_map(map<int,vector<vcg::Point2f>>& map, int key, vcg::Point2f value){
            if(map[key].size()==0 || (std::find(map[key].begin(), map[key].end(), value) == map[key].end())){
                map[key].push_back(value);
            }
        }
    
    public: 
        void add_value_to_map(map<long long, vector<vcg::Point2f>>& map, long long key, vector<vcg::Point2f> values){
            for(int i = 0; i < values.size(); i++){
                add_value_to_map(map, key, values[i]);
            }
        }

    public:
        void add_value_to_map(map<long long, vector<vcg::Point2f>>& map, long long key, vcg::Point2f value){
            
            vector<vcg::Point2f> values = map[key];

            if(values.size()==0 || (std::find(values.begin(), values.end(), value) == values.end())){
                map[key].push_back(value);
            }
        }

    public:
        void add_value_to_map(map<int, vector<long long>>& map, int key, vector<long long> values){
            for(int i = 0; i < values.size(); i++){
                add_value_to_map(map, key, values[i]);
            }
        }
    
    public:
        void add_value_to_map(map<int, vector<long long>>& map, int key, long long value){
            vector<long long> values = map[key];
            if(values.size()==0 || (std::find(values.begin(), values.end(), value) == values.end())){
                map[key].push_back(value);
                cout<<map[key].size()<<endl;
            }
        }

     
    /*
        @param map<int, map<long long, vector<vcg::Point2f>>>& map
        @param int key, the id of the vertex
        @param map<long long, vector<vcg::Point2f>>& value_map the values to add
      
        @description 
      
        @returns 
      
    */
    public:
        void add_value_to_map(map<int, map<long long, vector<vcg::Point2f>>>& outter_map, int key, map<long long, vector<vcg::Point2f>>& value_map){
            map<long long, vector<vcg::Point2f>> inner_map = outter_map[key];
            if(inner_map.size()==0){
                outter_map[key] = value_map;
            }
            else{
                for(auto it = value_map.cbegin(); it != value_map.cend(); ++it)
                {
                    long long inner_key = it->first;
                    vector<vcg::Point2f> values = inner_map[inner_key];
                    add_value_to_map(inner_map, inner_key, values);       
                }
                
                outter_map[key] = inner_map;
            }
        }

    /*
        All the methods called print_map work the same way with sligltly different parameters

        @param

        @description
            Given a pointer to a map and 
            - a key if you want to print a specific value or 
            - not giving any key to print all the values in the map

        @return
    */
    public:
        void print_map(map<int,vector<vcg::Point2f>>& map, int key){
            vector<vcg::Point2f> values = map[key];
            cout << "Size of map with key: [" << key << "] is: " << map[key].size() << endl;
            cout << "Elements with key [" << key << "] are: " <<endl;
            for(int i = 0; i < values.size(); i++)
            { 
                std::cout << "("<<values[i][0] <<","<<values[i][1]<<")"  << endl;
            }
            cout<<endl;

        }
    public:
        void print_map(map<int,vector<long long>>& map, int key){
            vector<long long> values = map[key];
            cout << "Size of map with key: [" << key << "] is: " << map[key].size() << endl;
            cout << "Elements with key [" << key << "] are: " <<endl;
            for(int i = 0; i < values.size(); i++)
            { 
                std::cout <<values[i] << endl;
            }
            cout<<endl;

        }

    public:
        void print_map(map<int,vector<vcg::Point2f>>& map){
            for(auto it = map.cbegin(); it != map.cend(); ++it)
            {
                int key = it->first;
                print_map(map, key);
            }
        }

    public: 
        void print_map(map<long long, vector<vcg::Point2f>>& map){
            for(auto it = map.cbegin(); it != map.cend(); ++it)
            {
                long long key = it->first;
                print_map(map, key);
            }
        }

    public:
        void print_map(std::map<int, Eigen::Tensor<float, 1>> map){
            for(auto it = map.cbegin(); it != map.cend(); ++it)
            {
                int key = it->first;
                print_map(map, key);
            }    
        }
    
    public:
        void print_map(std::map<int, Eigen::Tensor<float, 1>> map , int key){
            Eigen::Tensor<float, 1> tensor = map[key];
            // Get the size of the tensor
            int size = tensor.size();

            // Print each element
            for (int i = 0; i < size; ++i) {
                std::cout << tensor(i) << " ";
            }
            std::cout << std::endl;
            
        }
    
    public: 
        void print_map(map<long long, vector<vcg::Point2f>>& map, long long key){
            vector<vcg::Point2f> values = map[key];
            cout << "   Size of map with key: [" << key << "] is: " << map[key].size() << endl;
            cout << "   Elements with key [" << key << "] are: "<<endl;
            for(int i = 0; i < values.size(); i++)
            { 
                std::cout << "("<<values[i][0] <<","<<values[i][1]<<")"  << endl;
            }
            cout<<endl;

        }

    public:
        void print_map(map<int, map<long long, vector<vcg::Point2f>>>& outter_map){
            for(auto it = outter_map.cbegin(); it != outter_map.cend(); ++it)
            {
                int key = it->first;
                map<long long, vector<vcg::Point2f>> inner_map = outter_map[key];
                cout << "Size of map with key: [" << key << "] is: " << outter_map[key].size() << endl;
                cout << "Elements with key [" << key << "] are: "<<endl;
                print_map(inner_map);
            }
        }


    public:
        void print_map(map<long long, map<int, vector<vcg::Point2f>>> outter_map, long long key, int inner_key){

            map<int, vector<vcg::Point2f>> inner_map = outter_map[key];
            vector<vcg::Point2f> vertexes = inner_map[inner_key];

            if (vertexes.size() > 0){
                for (int i = 0; i < vertexes.size(); i++){
                    cout << "| x: " << vertexes[i][0] << " | y: " << vertexes[i][1]<<" |" <<endl;
                }
            }
            else
                cout << "vertex size at: | "<<inner_key << " | " << key << " | is empty" << endl;
            

        }

    /*
        All the methods called map_to_json work the same way with slightly different parameters 

        @param

        @description
            Given a pointer to a map, a save path and a file_name, this function to convert the map to a JSON string
            and saves it to the save_path/filename.json

        @return
    */
    public:
        void map_to_json(const std::map<int, std::vector<Point2f>>& data, string save_path, string timestamp) {
            // Check if the directory already exists
            if (!filesystem::exists(save_path)) {
                // Create the directory
                if (filesystem::create_directory(save_path)) {
                    std::cout << "Directory created successfully: "<< save_path << std::endl;
                } else {
                    std::cerr << "Error: Failed to create directory: " << save_path << std::endl;
                }
            }

            std::ostringstream oss;
            oss << "{\n";

            bool first_vertex = true;
            for (const auto& vertex_entry : data) {
                if (!first_vertex) {
                    oss << ",\n";
                }
                first_vertex = false;

                int vertex_id = vertex_entry.first;
                const std::vector<Point2f>& pixel_coords = vertex_entry.second;

                oss << "    \"" << vertex_id << "\": {\"pixel_coords\":[";

                bool first_point = true;
                for (const auto& point : pixel_coords) {
                    if (!first_point) {
                        oss << ",";
                    }
                    first_point = false;

                    oss << "{\"x\":" << point.X() << ", \"y\":" << point.Y() << "}";
                }

                oss << "]}";
            }

            oss << "\n}";
            
            string full_path = save_path+"/"+timestamp+".json";
            // Save JSON string to file
            std::ofstream file(full_path);
            file << oss.str();
            file.close();

        }

        // Function to convert the map to a JSON string
        public:
            void map_to_json(const std::map<int, std::vector<long long>>& data, string save_path, string json_name) {
                // Check if the directory already exists
                if (!filesystem::exists(save_path)) {
                    // Create the directory
                    if (filesystem::create_directory(save_path)) {
                        std::cout << "Directory created successfully: "<< save_path << std::endl;
                    } else {
                        std::cerr << "Error: Failed to create directory: " << save_path << std::endl;
                    }
                }

                json j;
                for (const auto& [key, value] : data) {
                    j[std::to_string(key)] = value;
                }

                string full_path = save_path+"/"+json_name+".json";
                std::ofstream file(full_path);
                if (!file.is_open()) {
                    std::cerr << "Unable to open file: " << full_path << std::endl;
                    return;
                }
                file << j.dump(4); // Pretty-print with 4 spaces of indentation
                file.close();
                std::cout << "JSON saved to " << full_path << std::endl;

            }

        public:
            void map_to_json(const std::map<int, double>& data, string save_path, string json_name) {
                // Check if the directory already exists
                if (!filesystem::exists(save_path)) {
                    // Create the directory
                    if (filesystem::create_directory(save_path)) {
                        std::cout << "Directory created successfully: "<< save_path << std::endl;
                    } else {
                        std::cerr << "Error: Failed to create directory: " << save_path << std::endl;
                    }
                }

                json j;
                for (const auto& [key, value] : data) {
                    j[std::to_string(key)] = value;
                }
                
                string full_path = save_path+"/"+json_name+".json";
                std::ofstream file(full_path);
                if (!file.is_open()) {
                    std::cerr << "Unable to open file: " << full_path << std::endl;
                    return;
                }
                file << j.dump(4); // Pretty-print with 4 spaces of indentation
                file.close();
                std::cout << "JSON saved to " << full_path << std::endl;

            }


    /*
        @param string save_path, path to the saved json
        @param string timestamp, json filename

        @description
            Given the path to the json file, returns a map of the data in the json

        @return std::map<int, std::vector<Point2f>> resultMap, the values of the json as a map
    */
    public:
        std::map<int, std::vector<Point2f>> json_to_map(string save_path, string timestamp){
            std::map<int, std::vector<Point2f>> resultMap;

            // Open the JSON file
            ifstream file(save_path+"/"+timestamp+".json");
            if (!file.is_open()) {
                cerr << "Failed to open file\n";
                throw std::invalid_argument( "received non existing path: "+save_path+"/"+timestamp+".json" );
            }       
            
            json jsonData;
            file >> jsonData;
            file.close();

            // Parse the JSON data
            for (auto& [key, value] : jsonData.items()) {
                int intKey = std::stoi(key);
                std::vector<Point2f> coords;
                
                for (auto& coord : value["pixel_coords"]) {
                    vcg::Point2f p2f_coord(coord["x"], coord["y"]);
                    //float x = coord["x"];
                    //float y = coord["y"];
                    coords.emplace_back(p2f_coord);
                }

                resultMap[intKey] = coords;
            }

            return resultMap;
        }

    /*
        @param const Eigen::Tensor<float, 3>& tensor, pointer to the 3D Eigen::tensor to save
        @param string save_path, path to the folder to save the .bin file
        @parma string file_name, name of the file to save

        @description
            given a Eigen::Tensor<float, 3>, saves it's values in a binary file

        @return
    */
    public:    
        void save_tensor_to_binary(const Eigen::Tensor<float, 3>& tensor, string save_path, string file_name) {
            // Check if the directory already exists
            if (!filesystem::exists(save_path)) {
                // Create the directory
                if (filesystem::create_directory(save_path)) {
                    std::cout << "Directory created successfully: "<< save_path << std::endl;
                } else {
                    std::cerr << "Error: Failed to create directory: " << save_path << std::endl;
                }
            }
            
            string full_path = save_path+"/"+file_name;
            
            std::ofstream file(full_path, std::ios::binary);
            if (!file) {
                std::cerr << "Unable to open file: " << full_path << std::endl;
                return;
            }

            // Save the dimensions of the tensor
            int dim0 = tensor.dimension(0);
            int dim1 = tensor.dimension(1);
            int dim2 = tensor.dimension(2);
            file.write(reinterpret_cast<const char*>(&dim0), sizeof(dim0));
            file.write(reinterpret_cast<const char*>(&dim1), sizeof(dim1));
            file.write(reinterpret_cast<const char*>(&dim2), sizeof(dim2));

            // Save the tensor data
            file.write(reinterpret_cast<const char*>(tensor.data()), dim0 * dim1 * dim2 * sizeof(float));

            file.close();
        }

    public:    
        void save_tensor_to_binary(const Eigen::Tensor<float, 1>& tensor, string save_path, string file_name) {
            // Check if the directory already exists
            if (!filesystem::exists(save_path)) {
                // Create the directory
                if (filesystem::create_directory(save_path)) {
                    std::cout << "Directory created successfully: "<< save_path << std::endl;
                } else {
                    std::cerr << "Error: Failed to create directory: " << save_path << std::endl;
                }
            }
            
            string full_path = save_path+"/"+file_name;
            
            std::ofstream file(full_path, std::ios::binary);
            if (!file) {
                std::cerr << "Unable to open file: " << full_path << std::endl;
                return;
            }

            // Save the dimensions of the tensor
            int dim0 = tensor.dimension(0);
            file.write(reinterpret_cast<const char*>(&dim0), sizeof(dim0));

            // Save the tensor data
            file.write(reinterpret_cast<const char*>(tensor.data()), dim0 * sizeof(float));

            file.close();
        }

    /*
        @param const HandleMesh& mesh_handler,
        @param const cv::Mat& depthImage,
        @param const Eigen::Matrix4d& extrinsic,
        @param const Eigen::Matrix3d& intrinsic,
        @param long long timestamp,
        @param string save_path,
        @param map<int,vector<long long>>& map_vertex_to_timestamp,
        @param float depthScale = 5000,
        @param float depth_threshold = 0.001,
        @param bool save_json = true

        @description
            This method is to map the 3D vertices of a given mesh to their corresponding 2D pixel coordinates in a 
            depth image, using the provided extrinsic and intrinsic camera parameters. It also associates each 
            vertex with a timestamp and optionally saves the resulting mapping to a JSON file.

        @return
            map<int, vector<vcg::Point2f>> vertex_map A map where the key is the vertex index and the value is a vector of 2D pixel coordinates (vcg::Point2f).
    */
    public: 
        map<int, vector<vcg::Point2f>> compute_vertexes_per_image(const HandleMesh& mesh_handler, const cv::Mat& depthImage, const Eigen::Matrix4d& extrinsic, const Eigen::Matrix3d& intrinsic, long long timestamp, string save_path, map<int,vector<long long>>& map_vertex_to_timestamp,  bool save_json = true, float depthScale = 5000, float depth_threshold = 0.001){
            map<int, vector<vcg::Point2f>> vertex_map;
            Eigen::Matrix4d extrinsicInverse = extrinsic.inverse();
            for (int vIdx = 0; vIdx < mesh_handler.mesh.vert.size(); ++vIdx) {
                vcg::Point3f vertex = mesh_handler.mesh.vert[vIdx].P();
                Eigen::Vector4d vertexHomogeneous(vertex[0], vertex[1], vertex[2], 1.0);
                Eigen::Vector4d camCoords = extrinsicInverse * vertexHomogeneous;

                if (camCoords[2] > 0) continue; 

                Eigen::Vector3d imageCoords = intrinsic * camCoords.head<3>();

                int x = std::round(imageCoords[0] / imageCoords[2]);
                int y = std::round(imageCoords[1] / imageCoords[2]);

                vcg::Point2f pixel(x, y);
                
                if (pixel[0] >= 0 && pixel[0] < depthImage.cols && pixel[1] >= 0 && pixel[1] < depthImage.rows) {
                    uint16_t depthValue = depthImage.at<uint16_t>(y, x);
                    float depth = depthValue / depthScale;

                    if(depth + depth_threshold > camCoords[2]){
                        vertex_map[vIdx].push_back(pixel);
                        #pragma omp critical
                        map_vertex_to_timestamp[vIdx].push_back(timestamp);
                        //cout << vIdx << " "<< timestamp << endl;
                        //add_value_to_map(map_vertex_to_timestamp, vIdx, timestamp);
                    }
                }
            }


            if (vertex_map.size()>0 && save_json)
                map_to_json(vertex_map, save_path, to_string(timestamp));
            
            return vertex_map;
            
        }

    /*
        @param const filesystem::path& directory, the path where the file is located
        @param const std::string& extension, the extension of the file to search

        @description this method returns a vector of paths to all the files with the given extention

        @return files vector of paths to the files with given extention
    */
    public:
        std::vector<filesystem::path> findFilesWithExtension(const filesystem::path& directory, const std::string& extension) {
            std::vector<filesystem::path> files;
            
            if (!filesystem::exists(directory) || !filesystem::is_directory(directory)) {
                throw std::runtime_error("Directory does not exist or is not a directory");
            }
            
            for (const auto& entry : filesystem::directory_iterator(directory)) {
                if (entry.is_regular_file() && entry.path().extension() == extension) {
                    files.push_back(entry.path());
                }
            }
            
            return files;
        }

    /*
        @param const std::string& file_path, the path to the .bin file
        @param int dim1=1080, the first dimension of the tensor
        @param int dim2=1920, the second dimension of the tensor
        @param int dim3=1024, the third dimension of the tensor

        @description given a path to a binary file, this method reads the file and adds its values to the Eigen::Tensor

        @return Eigen::TensorMap<Eigen::Tensor<float, 3>> tensor
    */
    public:
        Eigen::Tensor<float, 3> load_tensor_from_binary(const std::string& file_path, int dim1=1080, int dim2=1920, int dim3=1024) {
            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                std::cerr << "Unable to open file: " << file_path << std::endl;
                exit(EXIT_FAILURE);
            }

            std::vector<float> buffer(dim1 * dim2 * dim3);
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
            file.close();

            Eigen::TensorMap<Eigen::Tensor<float, 3>> tensor(buffer.data(), dim1, dim2, dim3);
            return tensor;
        }
    
    public:
        Eigen::Tensor<float, 1> load_tensor_from_binary(const std::string& file_path, bool oneD_tensor, int dim1=1024) {
            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                std::cerr << "Unable to open file: " << file_path << std::endl;
                exit(EXIT_FAILURE);
            }

            std::vector<float> buffer(dim1);
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
            file.close();

            Eigen::TensorMap<Eigen::Tensor<float, 1>> tensor(buffer.data(), dim1);
            return tensor;
        }
    

    public:
        std::map<int, Eigen::Tensor<float, 1>> load_all_tensors_from_bin(const std::string& file_path){

            std::map<int, Eigen::Tensor<float, 1>> tensors;
            auto bin_paths = findFilesWithExtension(file_path, ".bin");
            vector<string> feats_paths;
            for (const auto& file : bin_paths) {
                feats_paths.push_back(file.filename().stem().string());
                //std::cout << file.filename().stem().string() << std::endl;
            }    

            cout << "Loading all .bin files in: "<<file_path<<endl;            
            float done_load = 0;
            cout<<" Loaded "<<std::setprecision(3) << std::fixed<< done_load/feats_paths.size()*100 << "% | "<<static_cast<int>(done_load)<<"/"<<feats_paths.size()<<"\r" << std::flush;             

            #pragma omp parallel for 
            for(int i = 0; i < feats_paths.size(); i++){
                int key = stoi(feats_paths[i]);
                tensors[key] = load_tensor_from_binary(file_path+"/"+feats_paths[i]+".bin", true);

                done_load++;
                #pragma omp critical
                if (static_cast<int>(done_load) % static_cast<int>(feats_paths.size()/12) == 0)
                    cout<<" Loaded "<<std::setprecision(3) << std::fixed<< done_load/feats_paths.size()*100 << "% | "<<static_cast<int>(done_load)<<"/"<<feats_paths.size()<<"\r" << std::flush;             
            }

            cout<<" Loaded "<<std::setprecision(3) << 100 << "% | "<<feats_paths.size()<<"/"<<feats_paths.size()<<endl;             

            return tensors;
        }

    /*
        @param const Eigen::Tensor<float, 3>& tensor,
        @param int x, 
        @param int y,

        @description given a 3 dimensional tensor and two coordinates, x and y, 
            this method returns a one dimensional tensor containing all the values over the third dimension

        @return Eigen::Tensor<float, 1> values
    */
    Eigen::Tensor<float, 1> get_values_from_coordinates(const Eigen::Tensor<float, 3>& tensor, int x, int y) {
        int dim3 = tensor.dimension(2);
        Eigen::Tensor<float, 1> values(dim3);
        values.setZero();
        
        for (int z = 0; z < dim3; ++z) {
            //if (tensor(x, y, z) != 0){
            values(z) = tensor(x, y, z);
            //}
        }

        return values;
    }
    
    /*
        @param map<long long, map<int, vector<vcg::Point2f>>>& map_vertex, a map indexed by the frame timestamp of a map indexed by the vertex id of 2d coordinates
        @param std::vector<string> timestamps, the vector of all the timestamps of the .bin to concatenate
        @param string path_to_features, path to the folder containing the features extrated by clip
        @param string path_to_json, path to the json containing the vertex coornd map (not used for now) 
        
        @description Given the vector of timestamps and the map of features, for each timestamp, load the feature binary 
            file and
            - For each vertex, I retrieve the features (the 1x1x1024 tensor) for each pixel identified earlier.
            - I sum the features together (based on the vertex ID).
            - I normalize after summing all the features.
        
        @return std::map<int, Eigen::Tensor<float, 1>> tensors; 
    */
    public:
        std::map<int, Eigen::Tensor<float, 1>> concatenate_features(map<long long, map<int, vector<vcg::Point2f>>>& map_vertex, std::vector<string> timestamps, string path_to_features="./resources/dataset/cnr_c60/saved-feat", string path_to_json="./resources/dataset/cnr_c60/vertex_images_json"){
        
            int done_timestamps = 0;
            std::map<int, Eigen::Tensor<float, 1>> tensors;
            
            cout<<"\tLoading and summing tensors..."<<endl;
            //cout << "number of threads "<< n_threads << endl;
            cout<<"\t Processed "<<std::setprecision(3) << std::fixed<< done_timestamps/timestamps.size()*100 << "% | "<<static_cast<int>(done_timestamps)<<"/"<<timestamps.size()<<"\r" << std::flush;             
           
            for (int i = 0; i<timestamps.size(); i++){//timestamps.size()
                cout<<"\t\t\t"<< i<<"/"<<timestamps.size()<<": concatenate_features: " <<timestamps[i]<<"\r" << std::flush;
                auto tensor = load_tensor_from_binary(path_to_features+"/"+timestamps[i]+".bin");
                std::map<int, std::vector<Point2f>> vertex_image_json = map_vertex[stoll(timestamps[i])]; //json_to_map(path_to_json, timestamps[i]);
                
                for(auto it = vertex_image_json.cbegin(); it != vertex_image_json.cend(); ++it){
                    int key = it->first;   
                    std::vector<Point2f> p2f_json = vertex_image_json[key];
                    
                    #pragma omp parallel for 
                    for(int i = 0; i < p2f_json.size(); i++){
                        if(tensors[key].size()==0)
                            tensors[key] = get_values_from_coordinates(tensor, p2f_json[0][0], p2f_json[0][1]);
                        else
                            tensors[key] += get_values_from_coordinates(tensor, p2f_json[0][0], p2f_json[0][1]);
                    }
                }


                //print_map(tensors);
                done_timestamps++;
                std::cout<<"\t\tProcessed "<<std::setprecision(3) << std::fixed<< done_timestamps/timestamps.size()*100 << "% | "<<static_cast<int>(done_timestamps)<<"/"<<timestamps.size()<<"\r" << std::flush;
            }

            cout<<"\tNormalizing tensors.."<<endl;
            for(auto it = tensors.cbegin(); it != tensors.cend(); ++it){
                int key = it->first;   
                l2_normalization(tensors[key]);
            }

            return tensors;
        }

    public:
        void l2_normalization(Eigen::Tensor<float, 1>& tensor){
            // Compute the L2 norm of the tensor
            Eigen::Tensor<float, 0> norm = tensor.square().sum().sqrt();

            // Perform L2 normalization
            if(norm(0) > 0) { // To avoid division by zero
                tensor = tensor / norm(0);
            }
        }

    public:
        void minmax_normalization(Eigen::Tensor<float, 1>& tensor){

            Eigen::Tensor<float, 0> min_value = tensor.minimum();
            Eigen::Tensor<float, 0> max_value = tensor.maximum();

            // Compute the normalized tensor
            tensor = (tensor - min_value(0)) / (max_value(0) - min_value(0));
        }

    /*
        @param const Eigen::Tensor<float, 3>& tensor

        @description A support method to print a 3 dimensional Eigen::Tensor
    */
    public:
        void print_tensor(const Eigen::Tensor<float, 3>& tensor) {
            int dim0 = tensor.dimension(0);
            int dim1 = tensor.dimension(1);
            int dim2 = tensor.dimension(2);

            for (int i = 0; i < dim0; ++i) {
                for (int j = 0; j < dim1; ++j) {
                    for (int k = 0; k < 10; ++k) {
                        std::cout << tensor(i, j, k) << " " << endl;
                    }
                    std::cout << std::endl;  // Print a new line at the end of each row
                }
                std::cout << std::endl;  // Print a new line at the end of each matrix
            }
        }

    /*
        @param std::map<int, Eigen::Tensor<float, 1>>& tensors, the map of all the features indexed by the vertexes id
        @param map<long long, map<int, vector<vcg::Point2f>>>& vertex_map, a map indexed by the frame timestamp of a map indexed by the vertex id of 2d coordinates
        @param long long timestamp, timestamps of the .bin to save
        @param string save_path, the path to the save folder
        @param int dim1 = 1080, int dim2 = 1920, int dim3 = 1024, the dimensions of the 3d tensor

        @description
            Given a timestamp and the map of features, create a Eigen::Tensor<float,3> tensor with the dimensions 
    */
    public:
        void save_tensor_ordered(std::map<int, Eigen::Tensor<float, 1>>& tensors, map<long long, map<int, vector<vcg::Point2f>>>& vertex_map, long long timestamp, string save_path, int dim1 = 1080, int dim2 = 1920, int dim3 = 1024){
            
            map<int, vector<vcg::Point2f>> vertex = vertex_map[timestamp];
            //cout << "timestamp " << timestamp << endl;
            Eigen::Tensor<float,3> tensor(dim1, dim2, dim3);
            tensor.setZero();
            for(auto it = vertex.cbegin(); it != vertex.cend(); ++it){
                int key = it->first;   
                //cout << "key: " << key << endl;
                vector<vcg::Point2f> ver = vertex[key];
                vcg::Point2f coords = vertex[key][0];       
                int x = coords[0];
                int y = coords[1];
                //cout << "x " << x << " y " << y << endl;
                
                for(int i = 0; i < dim3; i++){
                    tensor(x,y,i) = tensors[key](i);
                }            
            }

            string ts = to_string(timestamp)+".bin";
            save_tensor_to_binary(tensor, save_path, ts);
        }        

    /*
        @param const std::string& file_name, path to file 
        @param Eigen::Tensor<float, 3>& tensor, pointer to a 3d tensor 

        @description 
            Given a path to a binary file containing a tensor, deserialize the data into a 3d Eigen::tensor of float
    */
    public:
        void read_tensor_from_binary(const std::string& file_name, Eigen::Tensor<float, 3>& tensor) {
            std::ifstream file(file_name, std::ios::binary);
            if (!file) {
                std::cerr << "Unable to open file: " << file_name << std::endl;
                return;
            }

            // Read the dimensions of the tensor
            int dim0, dim1, dim2;
            file.read(reinterpret_cast<char*>(&dim0), sizeof(dim0));
            file.read(reinterpret_cast<char*>(&dim1), sizeof(dim1));
            file.read(reinterpret_cast<char*>(&dim2), sizeof(dim2));

            // Resize the tensor to match the dimensions
            tensor.resize(dim0, dim1, dim2);

            // Read the tensor data
            file.read(reinterpret_cast<char*>(tensor.data()), dim0 * dim1 * dim2 * sizeof(float));
            file.close();
        }

        public:
            void read_tensor_from_binary(const std::string& file_name, Eigen::Tensor<float, 1>& tensor) {
                std::ifstream file(file_name, std::ios::binary);
                if (!file) {
                    std::cerr << "Unable to open file: " << file_name << std::endl;
                    return;
                }

                // Read the dimensions of the tensor
                int dim0;
                // Resize the tensor to match the dimensions
                tensor.resize(dim0);

                // Read the tensor data
                file.read(reinterpret_cast<char*>(tensor.data()), dim0 * sizeof(float));
                file.close();
            }


    /*

        @param string path_to_pv, path to raw data from hololens
        @param string path_to_depth_folder, path to the depth pics reconstructed using build_depth.cpp/.h
        @param string json_save_path, path to save json_files

        @description
            The following method works as follows:
            - For each vertex, I identify which pixels from which images compose it.
            
    */
    public:
        auto get_vetex_to_pixel_dict(string path_to_pv, string path_to_depth_folder, string clip_feat_path, string json_save_path,  bool save_json = true){
            cout << "Making map of vertex per timestamp.."<<endl;
            Project_point projector = Project_point(1);
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, 1, verbose);
            path_to_pv = path_to_dataset+path_to_pv;
            auto tuple_intrinsics = projector.extract_intrinsics(path_to_pv);
            float done_images = 0;
            map<long long, map<int, vector<vcg::Point2f>>> map_vertex;
            auto start = high_resolution_clock::now();

            std::cout<<"\tProcessed "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;

            auto bin_paths = findFilesWithExtension(path_to_dataset+clip_feat_path, ".bin");
            vector<string> bin_timestamp;
            for (const auto& file : bin_paths) {
                bin_timestamp.push_back(file.filename().stem().string());
                //std::cout << file.filename().stem().string() << std::endl;
            }

            map<int,vector<long long>> map_vertex_to_timestamp;

            #pragma omp parallel for 
            for (int i = 0; i < tuple_intrinsics.size(); i+=1){ //tuple_intrinsics.size()
                long long timestamp = std::get<0>(tuple_intrinsics[i]);
                
                if(std::find(bin_timestamp.begin(), bin_timestamp.end(), to_string(timestamp)) != bin_timestamp.end())
                {
                    Eigen::Matrix3d intrinsic = std::get<1>(tuple_intrinsics[i]);
                    Eigen::Matrix4d extrinsic = std::get<2>(tuple_intrinsics[i]);
                    
                    //cout<<to_string(timestamp)<<endl;
                    string path_to_depth = path_to_dataset+path_to_depth_folder+"/"+to_string(timestamp)+"_depth.png";
                    map_vertex[timestamp] = compute_vertexes_per_image(mesh_handler, loadDepthImage(path_to_depth), extrinsic, intrinsic, timestamp, path_to_dataset+json_save_path, map_vertex_to_timestamp, save_json);
                }
                
                done_images++;
                #pragma omp critical
                std::cout<<"\tProcessed "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;
            }

            auto end = high_resolution_clock::now();
            duration<double> elapsed = end - start;

            cout << endl << "Took " << elapsed.count() << " seconds" << endl;       
            cout<<""<<endl;

            return map_vertex;

        }
    
    /*

        @param map<long long, map<int, vector<vcg::Point2f>>> 
        @param string bin_save_path, path to save the tensors as .bin files

        @description
            The following method works as follows:
            - For each vertex, I retrieve the features (the 1x1x1024 tensor) for each pixel identified earlier.
            - I sum the features together (based on the vertex ID).
            - I normalize, using a l2-normalization, after summing all the features.
            - I reconstruct the image by setting a tensor to zero (1080x1920x1024) and replacing the pixels at 
                position (nxm) with the features we calculated (where the positions are dictated by the coordinates 
                based on the vertices).
    */
    public:
        std::map<int, Eigen::Tensor<float, 1>> make_tensors(map<long long, map<int, vector<vcg::Point2f>>> map_vertex, string clip_feat_path, string bin_save_path, bool save_tensors_per_ts = true, bool save_single_tensors = true){
            cout<<"Building tensors.."<<endl;
            auto start = high_resolution_clock::now();
            auto bin_paths = findFilesWithExtension(path_to_dataset+clip_feat_path, ".bin");
            vector<string> bin_timestamp;
            for (const auto& file : bin_paths) {
                bin_timestamp.push_back(file.filename().stem().string());
                //std::cout << file.filename().stem().string() << std::endl;
            }

            std::map<int, Eigen::Tensor<float, 1>> tensors = concatenate_features(map_vertex, bin_timestamp);
            
            if(save_single_tensors){
                cout<<"\tSaving single tensors.."<<endl;
                int done_images=0;
                std::cout<<"\t\tSaved "<<std::setprecision(3) << std::fixed<< done_images/tensors.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tensors.size()<<"\r" << std::flush;
                vector<int> all_keys;

                for(auto it = tensors.cbegin(); it != tensors.cend(); ++it){
                    int key = it->first;   
                    all_keys.push_back(key);
                }

                #pragma omp parallel for
                for(int i = 0; i < all_keys.size(); i++){
                    int key = all_keys[i];
                    save_tensor_to_binary(tensors[key], path_to_dataset+bin_save_path+"/single_vertex", to_string(key)+".bin");
                    #pragma omp critical
                    cout<<"\t\tSaved: " <<to_string(key)<<" "<<std::setprecision(3) << std::fixed<< done_images/tensors.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tensors.size()<<"\r" << std::flush;
                    done_images++;
                }                

            }


            if (save_tensors_per_ts){
                cout<<"\tSaving tensors per timestamp.."<<endl;
                int done_images=0;
                std::cout<<"\t\tSaved "<<std::setprecision(3) << std::fixed<< done_images/bin_timestamp.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<bin_timestamp.size()<<"\r" << std::flush;
                
                #pragma omp parallel for
                for (int i = 0; i < bin_timestamp.size(); i+=1){//tuple_intrinsics.size()
                    long long timestamp = stoll(bin_timestamp[i]);
                    save_tensor_ordered(tensors, map_vertex, timestamp, path_to_dataset+bin_save_path);
                    #pragma omp critical
                    cout<<"\t\tSaved: " <<to_string(timestamp)<<" "<<std::setprecision(3) << std::fixed<< done_images/bin_timestamp.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<bin_timestamp.size()<<"\r" << std::flush;
                    done_images++;
                }

            }

            auto end = high_resolution_clock::now();
            duration<double> elapsed = end - start;

            cout << "Took " << elapsed.count() << " seconds" << endl;       
            cout<<""<<endl;

            return tensors;
        }
};