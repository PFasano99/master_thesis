#include "../build_depth/build_depth.h"
#include <map>
#include <utility>      // std::pair, std::make_pair
#include <nlohmann/json.hpp>
#include <chrono>

//#include <torch/torch.h>
//#include <torch/script.h>
//#include <boost/python.hpp>
//#include <Python.h>

//namespace py = boost::python;

using namespace std::chrono;
using json = nlohmann::json;
using namespace std;
using namespace vcg;
//using namespace boost::python;

class Project_vertex_to_image
{

    private: int n_threads = 32;
    private: bool verbose = false;
    private: string path_to_mesh;
    private: string path_to_dataset;

    public:
        Project_vertex_to_image(string mesh_path, string dataset_path, int threads = 4, bool isVerbose = false)
        {
            n_threads = threads;
            verbose = isVerbose;
            path_to_mesh = mesh_path;
            path_to_dataset = dataset_path;
        }

    public:
        cv::Mat loadDepthImage(const std::string& filePath) {
            cv::Mat depthImage = cv::imread(filePath, cv::IMREAD_UNCHANGED);
            if (depthImage.empty()) {
                throw std::runtime_error("Could not load depth image at path: "+ filePath);
            }
            return depthImage;
        }

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


        // Function to convert the map to a JSON string
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


    public:// Function to save the map to a binary file
        void save_map_to_binary(const std::map<int, Eigen::Tensor<float, 1>>& map, string save_path, string file_name) {
            
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
                exit(EXIT_FAILURE);
            }

            // Save the size of the map
            size_t map_size = map.size();
            file.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));

            // Save each key and its corresponding tensor
            for (const auto& pair : map) {
                int key = pair.first;
                const Eigen::Tensor<float, 1>& tensor = pair.second;

                // Save the key
                file.write(reinterpret_cast<const char*>(&key), sizeof(key));

                // Save the size of the tensor
                int tensor_size = tensor.size();
                file.write(reinterpret_cast<const char*>(&tensor_size), sizeof(tensor_size));

                // Save the tensor data
                file.write(reinterpret_cast<const char*>(tensor.data()), tensor_size * sizeof(float));
            }

            file.close();
        }

    public: 
        map<int, vector<vcg::Point2f>> compute_vertexes_per_image(const HandleMesh& mesh_handler, const cv::Mat& depthImage, const Eigen::Matrix4d& extrinsic, const Eigen::Matrix3d& intrinsic, long long timestamp, string save_path, map<int,vector<long long>>& map_vertex_to_timestamp, float depthScale = 5000, float depth_threshold = 0.001,  bool save_json = true){
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

    Eigen::Tensor<float, 1> get_values_from_coordinates(const Eigen::Tensor<float, 3>& tensor, int x, int y) {
        int dim3 = tensor.dimension(2);
        Eigen::Tensor<float, 1> values(dim3);
        
        for (int z = 0; z < dim3; ++z) {
            values(z) = tensor(x, y, z);
        }

        return values;
    }

    public:
        std::map<int, Eigen::Tensor<float, 1>> concatenate_features(map<long long, map<int, vector<vcg::Point2f>>>& map, std::vector<string> timestamps, string path_to_features="./resources/dataset/cnr_c60/saved-feat", string path_to_json="./resources/dataset/cnr_c60/vertex_images_json",  string save_path = "", bool save_features = true){
        
            int done_timestamps = 0;
            std::map<int, Eigen::Tensor<float, 1>> tensors;

            cout<<"loading and summing tensors..."<<endl;
            std::cout<<"processed "<<std::setprecision(3) << std::fixed<< done_timestamps/timestamps.size()*100 << "% | "<<static_cast<int>(done_timestamps)<<"/"<<timestamps.size()<<"\r" << std::flush;

            omp_set_num_threads(n_threads);
            #pragma omp parallel for 
            for (int i = 0; i<timestamps.size(); i++){//timestamps.size()
                auto tensor = load_tensor_from_binary(path_to_features+"/"+timestamps[i]+".bin");
                std::map<int, std::vector<Point2f>> vertex_image_json = json_to_map(path_to_json, timestamps[i]);

                for(auto it = vertex_image_json.cbegin(); it != vertex_image_json.cend(); ++it){
                    int key = it->first;   
                    std::vector<Point2f> p2f_json = vertex_image_json[key];

                    for(int i = 0; i < p2f_json.size(); i++){
                        
                        if (tensors[key].size()==0){
                            #pragma omp critical
                            tensors[key] = get_values_from_coordinates(tensor, p2f_json[i][0], p2f_json[i][1]);        
                        }
                        else{
                            #pragma omp critical
                            tensors[key] += get_values_from_coordinates(tensor, p2f_json[i][0], p2f_json[i][1]); 
                        }

                    }
                }

                //print_map(tensors);

                done_timestamps++;
                #pragma omp critical
                std::cout<<"processed "<<std::setprecision(3) << std::fixed<< done_timestamps/timestamps.size()*100 << "% | "<<static_cast<int>(done_timestamps)<<"/"<<timestamps.size()<<"\r" << std::flush;

            }

            cout<<"normalizing tensors.."<<endl;
            
            for(auto it = tensors.cbegin(); it != tensors.cend(); ++it){
                int key = it->first;   

                Eigen::Tensor<float, 0> min_value = tensors[key].minimum();
                Eigen::Tensor<float, 0> max_value = tensors[key].maximum();

                // Compute the normalized tensor
                tensors[key] = (tensors[key] - min_value(0)) / (max_value(0) - min_value(0));   
            }
            
            return tensors;
        }
    
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
        auto get_vetex_to_pixel_dict(string path_to_pv, string path_to_depth_folder, string save_path){
            
            Project_point projector = Project_point(1);
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, 1, verbose);
            //
            path_to_pv = path_to_dataset+path_to_pv;
            auto tuple_intrinsics = projector.extract_intrinsics(path_to_pv);
            float done_images = 0;
            map<long long, map<int, vector<vcg::Point2f>>> map_vertex;
            auto start = high_resolution_clock::now();

            std::cout<<"processed "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;

            auto bin_paths = findFilesWithExtension(path_to_dataset+"cnr_c60/saved-feat", ".bin");
            vector<string> bin_timestamp;
            for (const auto& file : bin_paths) {
                bin_timestamp.push_back(file.filename().stem().string());
                //std::cout << file.filename().stem().string() << std::endl;
            }

            map<int,vector<long long>> map_vertex_to_timestamp;

            omp_set_num_threads(n_threads);
            #pragma omp parallel for 
            for (int i = 0; i < tuple_intrinsics.size(); i+=1){ //tuple_intrinsics.size()
                long long timestamp = std::get<0>(tuple_intrinsics[i]);
                
                if(std::find(bin_timestamp.begin(), bin_timestamp.end(), to_string(timestamp)) != bin_timestamp.end())
                {
                    Eigen::Matrix3d intrinsic = std::get<1>(tuple_intrinsics[i]);
                    Eigen::Matrix4d extrinsic = std::get<2>(tuple_intrinsics[i]);
                    
                    //cout<<to_string(timestamp)<<endl;
                    string path_to_depth = path_to_dataset+path_to_depth_folder+"/"+to_string(timestamp)+"_depth.png";
                    map_vertex[timestamp] = compute_vertexes_per_image(mesh_handler, loadDepthImage(path_to_depth), extrinsic, intrinsic, timestamp, path_to_dataset+save_path, map_vertex_to_timestamp);

                }
                
                #pragma omp critical
                std::cout<<"processed "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;
                done_images++;
            }

            cout<<"map_vertex_to_timestamp "<< map_vertex_to_timestamp.size()<<endl;
            map_to_json(map_vertex_to_timestamp, path_to_dataset+"/cnr_c60", "vertex_timestamp");
            std::map<int, Eigen::Tensor<float, 1>> tensors = concatenate_features(map_vertex, bin_timestamp);
            
            cout<<"saving tensors.."<<endl;
            done_images=0;
            std::cout<<"saved "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;

            omp_set_num_threads(n_threads);
            #pragma omp parallel for
            for (int i = 0; i < tuple_intrinsics.size(); i+=1){//tuple_intrinsics.size()
                long long timestamp = std::get<0>(tuple_intrinsics[i]);
                if(std::find(bin_timestamp.begin(), bin_timestamp.end(), to_string(timestamp)) != bin_timestamp.end())
                {
                    save_tensor_ordered(tensors, map_vertex, timestamp, path_to_dataset+"/cnr_c60/concat_feats");
                }

                #pragma omp critical
                std::cout<<"processed "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;
                done_images++;
            }

            auto end = high_resolution_clock::now();
            duration<double> elapsed = end - start;

            cout << " took " << elapsed.count() << " seconds" << endl;       
            cout<<""<<endl;
            //cout<<"map.size() "<<map.size()<<endl;
            return map_vertex;
        }
};