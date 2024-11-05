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
            omp_set_num_threads(n_threads);    
        }
    
    public:
        Project_vertex_to_image(int threads)
        {
            n_threads = threads;
            omp_set_num_threads(n_threads);    
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
        void loadDepthImage(const std::string& filePath, cv::Mat& depthImage, string timestamp) {
            depthImage = cv::imread(filePath, cv::IMREAD_ANYDEPTH);//cv::IMREAD_UNCHANGED
            if (depthImage.empty()) {
                throw std::runtime_error("Could not load depth image at path: "+ filePath);
            }
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
        void map_to_json(const std::map<int, float>& data, string save_path, string json_name) {
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
        void json_to_map(std::map<int, std::vector<Point2f>>& resultMap, string save_path, string timestamp){
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
            //cout<< "tensor dimension: " << dim0<<"x"<< dim1<<"x"<< dim2<<endl;
            //cout<< "dim0 * dim1 * dim2 * sizeof(float) " << dim0 * dim1 * dim2 * sizeof(float) << endl;
            //file.write(reinterpret_cast<const char*>(&dim0), sizeof(dim0));
            //file.write(reinterpret_cast<const char*>(&dim1), sizeof(dim1));
            //file.write(reinterpret_cast<const char*>(&dim2), sizeof(dim2));
            
            // Save the tensor data
            file.write(reinterpret_cast<const char*>(tensor.data()), dim0 * dim1 * dim2 * sizeof(float));

            file.close();
        }

    public:    
        void save_tensor_to_binary(const Eigen::Tensor<float, 2>& tensor, string save_path, string file_name) {
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
            cout<< "tensor dimension: " << dim0<<"x"<< dim1<<endl;
            //cout<< "dim0 * dim1 * dim2 * sizeof(float) " << dim0 * dim1 * dim2 * sizeof(float) << endl;
            //file.write(reinterpret_cast<const char*>(&dim0), sizeof(dim0));
            //file.write(reinterpret_cast<const char*>(&dim1), sizeof(dim1));

            // Save the tensor data
            file.write(reinterpret_cast<const char*>(tensor.data()), dim0 * dim1 * sizeof(float));

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
            //file.write(reinterpret_cast<const char*>(&dim0), sizeof(dim0));
            // Save the tensor data
            file.write(reinterpret_cast<const char*>(tensor.data()), dim0 * sizeof(float));

            //cout<< "tensor dimension: " << dim0<<endl;
            //cout<< "dim0 * sizeof(float) " << dim0 * sizeof(float) << endl;

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
        void compute_vertexes_per_image(map<int, vector<vcg::Point2f>>& vertex_map, const HandleMesh& mesh_handler, const cv::Mat& depthImage, const Eigen::Matrix4d& extrinsic, const Eigen::Matrix3d& intrinsic, long long timestamp, string save_path, map<int,vector<long long>>& map_vertex_to_timestamp,  bool save_json = true, float depthScale = 5000, const float depth_threshold = 0.01f){
            bool print = false;
            int test = 0;
            int test_2 = 0;
            int test_3 = 0;

            Eigen::Matrix3d dif_intrinsic = intrinsic;
            dif_intrinsic(0,2) = (static_cast<float>(depthImage.cols) - intrinsic(0,2));

            Eigen::Matrix4d extrinsicInverse = extrinsic.inverse();
            for (int vIdx = 0; vIdx < mesh_handler.mesh.vert.size(); ++vIdx) {
                vcg::Point3f vertex = mesh_handler.mesh.vert[vIdx].P();
                Eigen::Vector4d vertexHomogeneous(vertex[0], vertex[1], vertex[2], 1.0);
                Eigen::Vector4d camCoords = extrinsicInverse * vertexHomogeneous;

                if (camCoords[2] <= 0 || vIdx == 0) //we check for the z to be in front of the camera
                {
                    test++;
                    Eigen::Vector3d imageCoords = dif_intrinsic * camCoords.head<3>();
                    
                    int x = std::round(imageCoords[0] / imageCoords[2]);
                    int y = std::round(imageCoords[1] / imageCoords[2]);

                    /*
                        In computer graphics, the image coordinate system usually has its 
                        origin in the top-left corner, whereas in a typical camera or world
                        space, the origin is at the bottom-left. This can cause the points 
                        to appear vertically flipped when projected.
                    */
                    x = depthImage.cols - 1 - x; // Flip x-axis
                    
                    vcg::Point2f pixel(x, y);
                    if (pixel[0] >= 0 && pixel[0] < depthImage.cols && pixel[1] >= 0 && pixel[1] < depthImage.rows) {
                        test_2++;
                        /*
                            The depth image is being accessed with depthImage.at<float>(y, x) because 
                            cv::Mat uses row-major order where the first index is the row (y) and the 
                            second index is the column (x).
                        */
                        float depthValue = depthImage.at<float>(y,x);
                        if(depthValue + depth_threshold >= abs(camCoords[2]))//camCoords.head<3>().norm())
                        {
                            test_3++;
                            vertex_map[vIdx].push_back(pixel);
                        }
                    }
                } 
            }
            if (timestamp == 133468485245652555){
                cout << "number of vertex " << mesh_handler.mesh.vert.size() << endl;
                cout << "number of exepted vertex after first check "<< test << endl;
                cout << "number of exepted vertex after second check "<< test_2 << endl;
                cout << "number of exepted vertex after third check "<< test_3 << endl;
                cout << "vertex_map.size() "<<vertex_map.size()<<endl;
            }
                
            //cout << "vertex_map.size() "<< vertex_map.size() << endl; 
            if (vertex_map.size()>0 && save_json)
                map_to_json(vertex_map, save_path, to_string(timestamp));
                        
        }

    /*
        @param const filesystem::path& directory, the path where the file is located
        @param const std::string& extension, the extension of the file to search

        @description this method returns a vector of paths to all the files with the given extention

        @return files vector of paths to the files with given extention
    */
    public:
        void findFilesWithExtension(std::vector<filesystem::path>& files, const filesystem::path& directory, const std::string& extension) {            
            if (!filesystem::exists(directory) || !filesystem::is_directory(directory)) {
                throw std::runtime_error("Directory does not exist or is not a directory in findFilesWithExtension");
            }
            
            for (const auto& entry : filesystem::directory_iterator(directory)) {
                if (entry.is_regular_file() && entry.path().extension() == extension) {
                    files.push_back(entry.path());
                }
            }
            
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
        void load_tensor_from_binary(Eigen::Tensor<float, 3>& tensor, const std::string& file_path, int dim1=1080, int dim2=1920, int dim3=1024) {
            
            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                std::cerr << "Unable to open file: " << file_path << std::endl;
                exit(EXIT_FAILURE);
            }

            std::vector<float> buffer(dim1 * dim2 * dim3);
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
            file.close();

            //Eigen::TensorMap<Eigen::Tensor<float, 3>> tensor(buffer.data(), dim1, dim2, dim3);
            //tensor = Eigen::TensorMap<Eigen::Tensor<float, 3>>(buffer.data(), dim1, dim2, dim3);
            for(int r = 0; r < dim1; r++){
                for(int c = 0; c < dim2; c++){
                    for(int f = 0; f < dim3; f++){
                        int index = r * (dim2 * dim3) + c * dim3 + f;
                        tensor(r, c, f) = buffer[index];
                    }
                }
            }
        
        }
    
    public:
        void load_tensor_from_binary(Eigen::Tensor<float, 1>& tensor, const std::string& file_path, int dim1=1024) {
            std::ifstream file(file_path, std::ios::binary);
            if (!file) {
                std::cerr << "Unable to open file: " << file_path << std::endl;
                exit(EXIT_FAILURE);
            }

            std::vector<float> buffer(dim1);
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
            file.close();

            tensor = Eigen::TensorMap<Eigen::Tensor<float, 1>>(buffer.data(), dim1);
        }
    
    public:
        void load_all_tensors_from_bin(std::vector<Eigen::Tensor<float, 1>>& tensors, const std::string& file_path){
            std::vector<filesystem::path> bin_paths; 
            findFilesWithExtension(bin_paths, file_path, ".bin");
            vector<string> feats_paths;
            for (const auto& file : bin_paths) {
                feats_paths.push_back(file.filename().stem().string());
            }    

            cout << "Loading all .bin files in: "<<file_path<<endl;            
            float done_load = 0;
            cout<<" Loaded "<< done_load/feats_paths.size()*100 << "% | "<<static_cast<int>(done_load)<<"/"<<feats_paths.size()<<"\r" << std::flush;             

            #pragma omp parallel for 
            for(int i = 0; i < feats_paths.size(); i++){
                int key = stoi(feats_paths[i]);
                load_tensor_from_binary(tensors[key], file_path+"/"+feats_paths[i]+".bin");
                done_load++;
                
                if (static_cast<int>(done_load) % static_cast<int>(feats_paths.size()/10) == 0){
                    #pragma omp critical
                    cout<<" Loaded "<< (done_load/feats_paths.size())*100 << "% | "<<static_cast<int>(done_load)<<"/"<<feats_paths.size()<<"\r" << std::flush;             
                }
            }

            cout<<" Loaded "<< 100 << "% | "<<feats_paths.size()<<"/"<<feats_paths.size()<<endl;             
        }

    public:
        void load_all_tensors_from_bin(std::map<int, Eigen::Tensor<float, 1>>& tensors, const std::string& file_path){
            std::vector<filesystem::path> bin_paths; 
            
            findFilesWithExtension(bin_paths, file_path, ".bin");

            int cnt = 0;
            cout << "       Loaded: "<< cnt << "/" << bin_paths.size() << "\r" << std::flush;
            //omp_set_num_threads(n_threads);

            #pragma omp parallel for
            for(int i = 0; i < bin_paths.size(); i++){
                cnt++;
                string key = bin_paths[i].filename().stem().string();
                load_tensor_from_binary(tensors[stoi(key)], file_path+"/"+key+".bin");                
                
                if(cnt % 100000 == 0)
                {
                    #pragma omp critical
                    cout << "       Loaded: "<< cnt << "/" << bin_paths.size() << endl; //"\r" << std::flush;
                }
            }

            cout << "       Loaded: "<< cnt << "/" << bin_paths.size() << endl;
        }
    
    public:
        void load_all_tensors_from_bin(std::vector<Eigen::Tensor<float, 1>>& feature_map, const std::string& bin_file_path, int feature_size){
            // Open the binary file
            std::ifstream file(bin_file_path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Error: Unable to open file " << bin_file_path << std::endl;
            }

            // Calculate the size of the file
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);

            // Check if the file size is divisible by the feature size (i.e., consistent with the expected format)
            if (file_size % (feature_size * sizeof(float)) != 0) {
                std::cerr << "Error: File size is not divisible by feature size. Check the input file and feature size." << std::endl;
            }

            // Calculate the number of features (chunks) in the file
            int num_features = file_size / (feature_size * sizeof(float));
            // Prepare a buffer to read all the data at once
            std::vector<float> buffer(file_size / sizeof(float));
            file.read(reinterpret_cast<char*>(buffer.data()), file_size);
            
            // Close the file
            file.close();
            cout << "num_features " << num_features << endl;
            cout << "feature_map.size() " << feature_map.size() << endl;

            Eigen::Tensor<float, 1> values(1024); 
            values.setZero();
            feature_map.resize(num_features, values);


            // Iterate over the buffer and extract features
            for (int i = 0; i < num_features; ++i) {               
                // Create an Eigen::Tensor<float, 1> for each feature
                Eigen::Tensor<float, 1> feature_tensor(feature_size);
                
                // Copy the data from the buffer to the tensor
                for (int j = 0; j < feature_size; ++j) {
                    feature_tensor(j) = buffer[i * feature_size + j];
                }
                // Store the tensor in the map with the feature index as the key
                feature_map[i] = feature_tensor;
            }
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

    /*
        @param const Eigen::Tensor<float, 3>& tensor,
        @param int x, 
        @param int y,

        @description given a 3 dimensional tensor and two coordinates, x and y, 
            this method returns a one dimensional tensor containing all the values over the third dimension

        @return Eigen::Tensor<float, 1> values
    */
    public:
        Eigen::Tensor<float, 1> get_values_from_coordinates(const Eigen::Tensor<float, 3>& tensor, int x, int y) {
            int dim3 = tensor.dimension(2);

            Eigen::Tensor<float, 1> values(dim3);
            values.setZero();
            //values.setConstant(1.0f);
            
            for (int z = 0; z < dim3; ++z) {
                values(z) = static_cast<float>(tensor(y, x, z));
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
        void concatenate_features(std::vector<Eigen::Tensor<float, 1>>& tensors, map<long long, map<int, vector<vcg::Point2f>>>& map_vertex, std::vector<string> timestamps, bool normalize_tensors = true, string path_to_features="./resources/dataset/cnr_c60/saved-feat", string path_to_json="./resources/dataset/cnr_c60/vertex_images_json"){
            
            bool save_debug_mesh = false;

            HandleMesh mesh_handle = HandleMesh(path_to_mesh, 1, verbose);
            std::vector<float> n_iter(mesh_handle.mesh.vert.size(), 0);
            bool avg_normalize = false;
            vector<int> vertx_id;


            vector<vcg::Point3f> vertexes;
            if(save_debug_mesh)
            {
                vcg::Color4b white(255, 255, 255, 0);
                //set all vertex to color based on lower value 
                for(int i = 0; i < mesh_handle.mesh.vert.size(); i++){
                    mesh_handle.mesh.vert[i].C() = white;
                }

                Project_point projector = Project_point(n_threads, verbose);
                auto tuple_intrinsics = projector.extract_intrinsics(path_to_dataset+"dump_cnr_c60/");
            } 
                
            int done_timestamps = 0;            
            float completion_percentage = 0;

            cout<<" Loading and summing tensors..."<<endl;
            cout<<"     Processed "<< completion_percentage << "% | "<<static_cast<int>(done_timestamps)<<"/"<<timestamps.size()<<endl;          
            for (int i = 0; i<timestamps.size(); i++){//timestamps.size()
                completion_percentage = (done_timestamps/timestamps.size())*100;
                cout << "" << endl;
                std::cout<<"  Processing "<< completion_percentage << "% | "<<static_cast<int>(done_timestamps)<<"/"<<timestamps.size()<<": concatenate_features: " <<timestamps[i];
                Eigen::Tensor<float, 3> tensor(1080,1920,1024); 
                tensor.setZero();
                load_tensor_from_binary(tensor, path_to_features+"/"+timestamps[i]+".bin");
                
                cv::Mat image;
                cv::Mat image_2;
                if(save_debug_mesh){
                    image = cv::imread(path_to_dataset+"cnr_c60/open_clip_rgb/"+timestamps[i]+".png");
                    int width = 1920;
                    int height = 1080;
                    image_2 = cv::Mat::zeros(height, width, CV_8UC3); // 8-bit, 3-channel image (BGR)
                    //save_tensor_to_binary(tensor, path_to_dataset+"cnr_c60/concat_feats/", timestamps[i]+"_og.bin");
                    Eigen::Tensor<float,3> img_to_save; img_to_save.setZero();
                }
                
                std::map<int, std::vector<Point2f>> vertex_image_json = map_vertex[stoll(timestamps[i])]; //json_to_map(path_to_json, timestamps[i]);

                std::vector<int> keys;
                for(auto it = vertex_image_json.cbegin(); it != vertex_image_json.cend(); ++it){
                    keys.push_back(it->first);
                }

                //cout << "\nkeys.size() " << keys.size() << endl;
                #pragma omp parallel for
                for(int k = 0; k < keys.size(); k++){
                    int key = keys[k];   
                    vcg::Point2f p2f_json = vertex_image_json[key][0];
                    tensors[key] += get_values_from_coordinates(tensor, p2f_json[0], p2f_json[1]); 
                    n_iter[key] += 1;
                    
                    if(save_debug_mesh){
                        cv::Vec3b color = image.at<cv::Vec3b>(p2f_json[1], p2f_json[0]);
                        image_2.at<cv::Vec3b>(p2f_json[1], p2f_json[0]) = color;
                        vcg::Color4b nc(color[2], color[0], color[1], 0);
                        mesh_handle.mesh.vert[key].C() = nc;
                        #pragma omp critical
                        vertexes.push_back(mesh_handle.mesh.vert[key].P());
                    }  
                }    
                
                done_timestamps++;
                
                if(save_debug_mesh){
                    std::string filename = path_to_dataset+"cnr_c60/concat_feats/rgb_feats/"+timestamps[i]+"_red.png";
                    bool isSuccess = cv::imwrite(filename, image_2);

                    //Check if the image is saved successfully
                    if (isSuccess) {
                        std::cout << "Image saved successfully: " << filename << std::endl;
                    } else {
                        std::cout << "Error in saving the image" << std::endl;
                    }
                }                
            }

            cout<<"  Processed "<< completion_percentage << "% | "<<static_cast<int>(done_timestamps)<<"/"<<timestamps.size()<<endl;          

            if(save_debug_mesh){
                string path_to_save_mesh = path_to_dataset+"cnr_c60/concat_feats/proj_"+getTimestamp()+".ply";
                string path_to_save_mesh_vert = path_to_dataset+"cnr_c60/concat_feats/vert_"+getTimestamp()+".ply";
                int mask = vcg::tri::io::Mask::IOM_VERTCOORD;
                mask |= vcg::tri::io::Mask::IOM_VERTQUALITY;
                mask |= vcg::tri::io::Mask::IOM_VERTCOLOR;
                mesh_handle.save_mesh(path_to_save_mesh, mask);
                cout << "saved mesh at: " << path_to_save_mesh<< endl;
                mesh_handle.visualize_points_in_mesh(vertexes[0], vertexes, path_to_save_mesh_vert);    
            }
            
            if(avg_normalize){
                average_normalization(tensors, n_iter);
            }

            if(normalize_tensors){
                cout<<" Normalizing tensors.."<<endl;
                for(int k = 0; k < tensors.size(); k++){
                    l2_normalization(tensors[k]);
                }    
            }

        } 
    
    public:
        void average_normalization(std::vector<Eigen::Tensor<float, 1>>& tensors, std::vector<float>& n_iter){
            for(int k = 0; k < tensors.size(); k++){
                if (n_iter[k] != 0){
                    tensors[k] = tensors[k] / n_iter[k];
                }
            }
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
        void save_tensor_ordered(std::vector<Eigen::Tensor<float, 1>>& tensors, map<long long, map<int, vector<vcg::Point2f>>>& vertex_map, long long timestamp, string save_path, int dim1 = 1080, int dim2 = 1920, int dim3 = 1024){
            //cout << "tensors.size() " << tensors.size() << endl;
            map<int, vector<vcg::Point2f>> vertex = vertex_map[timestamp];  
            //cout << "timestamp " << timestamp << endl;
            Eigen::Tensor<float,3> tensor(dim1, dim2, dim3);
            //std::vector<std::vector<float>> matrix(1920, std::vector<float>(1080, 0.0f));

            tensor.setZero();
            for(auto it = vertex.cbegin(); it != vertex.cend(); ++it){
                int key = it->first;   
                //cout << "key: " << key << endl;
                vector<vcg::Point2f> ver = vertex[key];
                vcg::Point2f coords = vertex[key][0];       
                int x = coords[0];
                int y = coords[1];

                for(int i = 0; i < dim3; i++){
                    tensor(x,y,i) = static_cast<float>(tensors[key](i));
                    //matrix[x][y] += static_cast<float>(tensors[key](i));
                }     

            }

            string ts = to_string(timestamp)+"_0_1.bin";
            save_tensor_to_binary(tensor, save_path, ts);

            /*
            string filename = save_path+"/"+to_string(timestamp)+"_0_1.txt";
            std::ofstream outFile(filename);

            if (!outFile) {
                std::cerr << "Error opening file for writing: " << filename << std::endl;
                return;
            }

            // Write the matrix to the file
            for (const auto& row : matrix) {
                for (size_t col = 0; col < row.size(); ++col) {
                    outFile << row[col]; // Write the element
                    if (col < row.size() - 1) { // Add a space between elements
                        outFile << " ";
                    }
                }
                outFile << "\n"; // End of the row
            }

            outFile.close();
            std::cout << "Matrix saved to " << filename << std::endl;
            */

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
        void save_tensor_as_txt(Eigen::Tensor<float, 1> tensor, string save_path){
            // Open a file in write mode
            std::ofstream file(save_path);

            // Check if the file is open
            if (file.is_open()) {
                // Loop through the tensor and write to the file
                for (int i = 0; i < tensor.size(); ++i) {
                    file << tensor(i) << std::endl;  // Write each element followed by a newline
                }

                file.close();  // Close the file
                //std::cout << "Tensor successfully written to tensor_output.txt" << std::endl;
            } else {
                std::cerr << "Unable to open the file!" << std::endl;
            }
        }

    public:
        void get_vetex_to_pixel_dict(map<long long, map<int, vector<vcg::Point2f>>>& map_vertex,string path_to_pv, string path_to_depth_folder, string clip_feat_path, string json_save_path,  bool save_json = true){
            cout << "Making map of vertex per timestamp.."<<endl;
            Project_point projector = Project_point(1);
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, 1, verbose);
            path_to_pv = path_to_dataset+path_to_pv;
            auto tuple_intrinsics = projector.extract_intrinsics(path_to_pv);
            float done_images = 0;
            
            auto start = high_resolution_clock::now();


            std::vector<filesystem::path> bin_paths; 
            findFilesWithExtension(bin_paths, path_to_dataset+clip_feat_path, ".bin");
            vector<string> bin_timestamp;
            for (const auto& file : bin_paths) {
                bin_timestamp.push_back(file.filename().stem().string());
                //std::cout << file.filename().stem().string() << std::endl;
            }

            std::cout<<" Processed "<< (done_images/tuple_intrinsics.size())*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<endl;
            map<int,vector<long long>> map_vertex_to_timestamp;
            for (int i = 0; i < tuple_intrinsics.size(); i+=1){ //tuple_intrinsics.size()
                long long timestamp = std::get<0>(tuple_intrinsics[i]);
                done_images++;

                if(std::find(bin_timestamp.begin(), bin_timestamp.end(), to_string(timestamp)) != bin_timestamp.end())
                {
                    Eigen::Matrix3d intrinsic = std::get<1>(tuple_intrinsics[i]);
                    Eigen::Matrix4d extrinsic = std::get<2>(tuple_intrinsics[i]);
                    
                    //cout<<to_string(timestamp)<<endl;
                    string path_to_depth = path_to_dataset+path_to_depth_folder+"/"+to_string(timestamp)+"_depth.pfm";
                    cv::Mat depthImage;
                    loadDepthImage(path_to_depth, depthImage, to_string(timestamp));
                    compute_vertexes_per_image(map_vertex[timestamp], mesh_handler, depthImage, extrinsic, intrinsic, timestamp, path_to_dataset+json_save_path, map_vertex_to_timestamp, save_json);
                    std::cout<<" Processed "<< (done_images/tuple_intrinsics.size())*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;
                }        
            }

            std::cout<<" Processed "<< (done_images/tuple_intrinsics.size())*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;


            auto end = high_resolution_clock::now();
            duration<float> elapsed = end - start;

            cout << endl << "Took " << elapsed.count() << " seconds" << endl;       
            cout<<""<<endl;

        }
    
    public:
        void saveTensorsToBinary(const std::vector<Eigen::Tensor<float, 1>>& tensors, const std::string& filename) {
            std::ofstream outFile(filename, std::ios::binary);

            if (!outFile) {
                std::cerr << "Error opening file for writing: " << filename << std::endl;
                return;
            }

            // Save the number of tensors
            size_t numTensors = tensors.size();
            //outFile.write(reinterpret_cast<const char*>(&numTensors), sizeof(numTensors));

            // Iterate through each tensor and write its size and data
            for (const auto& tensor : tensors) {
                size_t tensorSize = tensor.size();

                // Write the tensor data
                outFile.write(reinterpret_cast<const char*>(tensor.data()), tensorSize * sizeof(float));
            }
            outFile.close();
        }

    public:
        void save_vertex_coords(map<long long, map<int, vector<vcg::Point2f>>> full_map, long long timestamp, string save_path){
            save_vertex_coords(full_map[timestamp], timestamp, save_path);
        }

    public:
        void save_vertex_coords(map<int, vector<vcg::Point2f>> vtx_coords, long long timestamp, string save_path){
            string filename = save_path+to_string(timestamp)+"_vtx_coords.txt";
            std::ofstream outfile(filename); // Open file for writing

            if (!outfile.is_open()) {
                std::cerr << "Error opening file for writing!" << std::endl;
                return;
            }

            for(auto it = vtx_coords.cbegin(); it != vtx_coords.cend(); ++it){
                int key = it->first;
                vector<vcg::Point2f> points = it->second;
                outfile << key << " ";
                outfile << points[0][0] << " " << points[0][1] << " ";
                outfile << std::endl; // End the line after each map entry
            }

            outfile.close(); // Close the file    
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
        std::vector<Eigen::Tensor<float, 1>> make_tensors(map<long long, map<int, vector<vcg::Point2f>>> map_vertex, string clip_feat_path, string bin_save_path, bool save_tensors_per_ts = true, bool save_single_tensors = true, bool normalize_tensor = true){
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, 1, verbose);
            
            cout<<"Building tensors.."<<endl;
            auto start = high_resolution_clock::now();
            std::vector<filesystem::path> bin_paths;
            findFilesWithExtension(bin_paths, clip_feat_path, ".bin");
            vector<string> bin_timestamp;
            for (const auto& file : bin_paths) {
                bin_timestamp.push_back(file.filename().stem().string());
                //std::cout << file.filename().stem().string() << std::endl;
            }

            Eigen::Tensor<float, 1> values(1024); 
            values.setZero();
            std::vector<Eigen::Tensor<float, 1>> tensors(mesh_handler.mesh.vert.size(), values);

            //void concatenate_features(std::vector<Eigen::Tensor<float, 1>>& tensors, map<long long, map<int, vector<vcg::Point2f>>>& map_vertex, std::vector<string> timestamps, bool normalize_tensors = true, string path_to_features="./resources/dataset/cnr_c60/saved-feat", string path_to_json="./resources/dataset/cnr_c60/vertex_images_json"){
            concatenate_features(tensors, map_vertex, bin_timestamp, normalize_tensor, clip_feat_path);
            saveTensorsToBinary(tensors, clip_feat_path+"/dinoV2_all_feats.bin");
            
            if(save_single_tensors){
                cout<<" Saving single tensors.."<<endl;
                int done_images=0;
                std::cout<<"  Saved "<< (done_images/tensors.size())*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tensors.size()<<endl;
                
                #pragma omp parallel for
                for(int i = 0; i < tensors.size(); i++){
                    Eigen::Tensor<float, 0> check_zero = tensors[i].sum();
                    if(check_zero(0) != 0.0f)
                        save_tensor_to_binary(tensors[i], path_to_dataset+bin_save_path+"/single_vertex", to_string(i)+".bin");
                    
                    if (done_images % tensors.size()/12 == 0){
                        #pragma omp critical
                        cout<<"  Saved: " <<static_cast<int>(done_images)<<"/"<<tensors.size()<<"\r" << std::flush;
                    }
                    done_images++;
                }                

            }


            if (save_tensors_per_ts){
                cout<<" Saving tensors per timestamp.."<<endl;
                int done_images=0;
                std::cout<<"  Saved "<< done_images/bin_timestamp.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<bin_timestamp.size()<<endl;
                
                #pragma omp parallel for
                for (int i = 0; i < bin_timestamp.size(); i+=1){//bin_timestamp.size()
                    long long timestamp = stoll(bin_timestamp[i]);
                    //if(timestamp == 133468485245652555)
                    if(i<1){
                        save_tensor_ordered(tensors, map_vertex, timestamp, path_to_dataset+bin_save_path);
                        
                        vector<Eigen::Tensor<float, 1>> ts_vertex;
                        for(int g = 0; g < tensors.size(); g++){
                            if(map_vertex[timestamp].find(g) != map_vertex[timestamp].end()){
                                ts_vertex.push_back(tensors[g]);
                            }
                        }
                        saveTensorsToBinary(ts_vertex, path_to_dataset+"cnr_c60/concat_feats/"+to_string(timestamp)+"_nz_feat.bin");

                    }
                    #pragma omp critical
                    cout<<"  Saved: " <<to_string(timestamp)<<" "<< (done_images/bin_timestamp.size())*100 << "% | "<<static_cast<int>(done_images)<<"/"<<bin_timestamp.size()<<"\r" << std::flush;
                    done_images++;
                }
                cout<<"\n  Saved: " << (done_images/bin_timestamp.size())*100 << "% | "<<static_cast<int>(done_images)<<"/"<<bin_timestamp.size()<<"\r" << std::flush;

            }

            auto end = high_resolution_clock::now();
            duration<float> elapsed = end - start;

            cout << "\nTook " << elapsed.count() << " seconds\n" << endl;       
            return tensors;
        }

    
    public:
        void color_mesh_by_features(string path_to_rgb_feat, string path_to_save_mesh, bool from_txt = false){
            HandleMesh mesh_handle = HandleMesh(path_to_mesh, 1, verbose);

            vcg::Color4b white(255, 255, 255, 0);
            //set all vertex to color based on lower value 
            for(int i = 0; i < mesh_handle.mesh.vert.size(); i++){
                mesh_handle.mesh.vert[i].C() = white;
            }

            if(from_txt){
                
                map<int, vcg::Color4b> vtx_col;
                
                for(int i = 0; i < mesh_handle.mesh.vert.size(); i++){
                    vtx_col[i] = white;
                }

                read_file_to_map(path_to_rgb_feat, vtx_col);
                cout << "vtx_col.size()" << vtx_col.size() << endl;
                
                for(auto it = vtx_col.cbegin(); it != vtx_col.cend(); ++it){
                    int key = it->first;
                    mesh_handle.mesh.vert[key].C() = vtx_col[key];
                    //cout << key << " "<< static_cast<int>(vtx_col[key][0]) << " " << static_cast<int>(vtx_col[key][1]) << " " << static_cast<int>(vtx_col[key][2]) << " " << endl;
                }

            }
            else{
                cv::Mat image = cv::imread(path_to_rgb_feat);//path_to_dataset+"cnr_c60/concat_feats/rgb_feats/3_"+timestamps[i]+"_og.png");
                int height = image.rows;  // Number of rows (height of the image)
                int width = image.cols;   // Number of columns (width of the image)
                int last_highest_id = 0;
                for(int h = 0; h < height; h++){
                    //cout << (h * height) << endl;
                    for(int w = 0; w < width; w++){
                        int key = last_highest_id + w ;
                        if(key < mesh_handle.mesh.vert.size()){
                            cv::Vec3b color = image.at<cv::Vec3b>(h, w);
                            vcg::Color4b nc(color[0], color[1], color[2], 0);
                            mesh_handle.mesh.vert[key].C() = nc;
                        }
                        else{
                            break;
                        }
                    }
                    last_highest_id += width;
                }
            }
            
            

            int mask = vcg::tri::io::Mask::IOM_VERTCOORD;
            mask |= vcg::tri::io::Mask::IOM_VERTQUALITY;
            mask |= vcg::tri::io::Mask::IOM_VERTCOLOR;
            mesh_handle.save_mesh(path_to_save_mesh, mask);
            cout << "saved mesh at: " << path_to_save_mesh<< endl;
        }


    public:    
    // Function to read the file and populate the map
        void read_file_to_map(const std::string &file_path, map<int, vcg::Color4b>& colour_map) {
            bool normalize_to_colour = false;
            std::ifstream file(file_path);
            std::string line;

            if (!file.is_open()) {
                std::cerr << "Error opening file!" << std::endl;
            }

            while (std::getline(file, line)) {
                std::istringstream iss(line);
                int id;

                // Parse the line, assuming the format is "id [r g b]"
                std::string id_str, rgb_values;
                if (std::getline(iss, id_str, ' ') && std::getline(iss, rgb_values)) {
                    id = std::stoi(id_str);  // Convert the first part to an integer ID

                    // Remove the brackets from the rgb_values string
                    rgb_values.erase(0, 1);  // Remove the first '['
                    rgb_values.erase(rgb_values.size() - 1);  // Remove the last ']'
                    //cout << rgb_values << endl;
                    // Parse the r, g, b values
                    std::istringstream rgb_stream(rgb_values);

                    float f1, f2, f3;
                    if (rgb_stream >> f1 >> f2 >> f3) {
                        // Convert to integers
                        //cout << f1 << " " << f2 << " " << f3 << endl; 
                        if(normalize_to_colour){
                            normalize_to_rgb(f1,f2,f3);
                        }
                        vcg::Color4b colour(f1, f2, f3, 0);
                        colour_map[id] = colour;
                    }
                    // Insert into the map

                } else {
                    std::cerr << "Error parsing line: " << line << std::endl;
                }
            }

            file.close();       
        }
    

    public:    
    // Function to read the file and populate the map
        void read_file_to_map(const std::string &file_path, map<int, vcg::Point2f>& coord_map) {
            std::ifstream file(file_path);
            std::string line;

            if (!file.is_open()) {
                std::cerr << "Error opening file!" << std::endl;
            }

            while (std::getline(file, line)) {
                std::istringstream iss(line);
                int id;

                // Parse the line, assuming the format is "id [r g b]"
                std::string id_str, coord_value;
                if (std::getline(iss, id_str, ' ') && std::getline(iss, coord_value)) {
                    id = std::stoi(id_str);  // Convert the first part to an integer ID

                    std::istringstream coord_stream(coord_value);

                    int f1, f2;
                    if (coord_stream >> f1 >> f2) {
                        // Convert to integers
                        //cout << f1 << " " << f2 << " " << f3 << endl; 
                        vcg::Point2f coord(f1,f2);
                        coord_map[id] = coord;
                    }
                    // Insert into the map

                } else {
                    std::cerr << "Error parsing line: " << line << std::endl;
                }
            }

            file.close();       
        }
    
    public:
        void normalize_to_rgb(float& x, float& y, float& z) {
            // Find the minimum and maximum values among the three floats
            float minValue = std::min({x, y, z});
            float maxValue = std::max({x, y, z});

            // Prevent division by zero if all values are the same
            if (minValue == maxValue) {
                x = y = z = 255.0f;  // All values map to 255 in this case
                return;
            }

            // Normalize the values between 0 and 255
            x = (x - minValue) / (maxValue - minValue) * 255.0f;
            y = (y - minValue) / (maxValue - minValue) * 255.0f;
            z = (z - minValue) / (maxValue - minValue) * 255.0f;
    }
};
