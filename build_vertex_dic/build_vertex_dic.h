#include "../build_depth/build_depth.h"
#include <map>

using namespace std;
using namespace vcg;

class Project_vertex_to_image
{

    private: int n_threads = 1;
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
        void add_value_to_map(map<vcg::Point2f,vector<vcg::Point3f>>& map, vcg::Point2f key, vcg::Point3f value){
            
            vector<vcg::Point3f> values = map[key];

            if(values.size()==0 || (std::find(values.begin(), values.end(), value) == values.end())){
                values.push_back(value);
                map[key] = values;
            }
        }
    
    public:
        void add_value_to_map(map<vcg::Point2f,vector<vcg::Point3f>>& map, vcg::Point2f key, vector<vcg::Point3f> values){
            for(int i = 0; i < values.size(); i++){
                add_value_to_map(map, key, values[i]);
            }
        }

    public: 
        void add_value_to_map(map<vcg::Point2f,vector<int>>& map, vcg::Point2f key, vector<int> values){
            for(int i = 0; i < values.size(); i++){
                add_value_to_map(map, key, values[i]);
            }
        }

    public:
        void add_value_to_map(map<vcg::Point2f,vector<int>>& map, vcg::Point2f key, int value){
            
            vector<int> values = map[key];

            if(values.size()==0 || (std::find(values.begin(), values.end(), value) == values.end())){
                values.push_back(value);
                map[key] = values;
            }
        }
    
    public: 
        void add_value_to_map(map<int,vector<vcg::Point2f>>& map, int key, vector<vcg::Point2f> values){
            for(int i = 0; i < values.size(); i++){
                add_value_to_map(map, key, values[i]);
            }
        }

    public:
        void add_value_to_map(map<int,vector<vcg::Point2f>>& map, int key, vcg::Point2f value){
            
            vector<vcg::Point2f> values = map[key];

            if(values.size()==0 || (std::find(values.begin(), values.end(), value) == values.end())){
                values.push_back(value);
                map[key] = values;
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
                values.push_back(value);
                map[key] = values;
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
        void print_map(map<vcg::Point2f,vector<vcg::Point3f>>& map, vcg::Point2f key){
            vector<vcg::Point3f> values = map[key];
            cout << "Size of map with key: [" << key[0]<<","<<key[1]<< "] is: " << map[key].size() << endl;
            cout << "Elements with key [" << key[0]<<","<<key[1]<< "] are: "<<endl;
            for(int i = 0; i < values.size(); i++)
            { 
                string element = "  ("+std::to_string(values[i][0])+" "+std::to_string(values[i][1])+" "+std::to_string(values[i][2])+")";
                std::cout << element  << endl;
            }
            cout<<endl;

        }

    public:
        void print_map(map<vcg::Point2f,vector<vcg::Point3f>>& map){
            for(auto it = map.cbegin(); it != map.cend(); ++it)
            {
                vcg::Point2f key = it->first;
                print_map(map, key);
            }
        }
    
    public:
        void print_map(map<vcg::Point2f,vector<int>>& map, vcg::Point2f key){
            vector<int> values = map[key];
            cout << "Size of map with key: [" << key[0]<<","<<key[1]<< "] is: " << map[key].size() << endl;
            cout << "Elements with key [" << key[0]<<","<<key[1]<< "] are: "<<endl;
            for(int i = 0; i < values.size(); i++)
            { 
                std::cout << values[i]  << endl;
            }
            cout<<endl;

        }

    public:
        void print_map(map<vcg::Point2f,vector<int>>& map){
            for(auto it = map.cbegin(); it != map.cend(); ++it)
            {
                vcg::Point2f key = it->first;
                print_map(map, key);
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


        // Function to convert the map to a JSON string
    public:
        void mapToJson(const std::map<int, std::vector<Point2f>>& data, string save_path, string timestamp) {
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

            oss << "}";
            
            string full_path = save_path+"/"+timestamp+".json";
            // Save JSON string to file
            std::ofstream file(full_path);
            file << oss.str();
            file.close();

        }


    public:
        void compute_vertexes_per_image(const HandleMesh& mesh_handler, const cv::Mat& depthImage, const Eigen::Matrix4d& extrinsic, const Eigen::Matrix3d& intrinsic){
            map<vcg::Point2f,vector<vcg::Point3f>> map;
            Eigen::Matrix4d extrinsicInverse = extrinsic.inverse();
            for (int vIdx = 0; vIdx < mesh_handler.mesh.vert.size(); ++vIdx) {
                vcg::Point3f vertex = mesh_handler.mesh.vert[vIdx].P();
                Eigen::Vector4d vertexHomogeneous(vertex[0], vertex[1], vertex[2], 1.0);
                Eigen::Vector4d camCoords = extrinsicInverse * vertexHomogeneous;
                
                if (camCoords[2] <= 0) continue; // Vertex is behind the camera

                Eigen::Vector3d imageCoords = intrinsic * camCoords.head<3>();
                cv::Point2f pixel(imageCoords[0] / imageCoords[2], imageCoords[1] / imageCoords[2]);

                if (pixel.x >= 0 && pixel.x < depthImage.cols && pixel.y >= 0 && pixel.y < depthImage.rows) {
                    vcg::Point2f key((int)pixel.x,(int)pixel.y);
                    add_value_to_map(map, key, vertex);
                }
            }

            //print_map(map);
            //cout<<"mesh_handler.mesh.vert.size() "<<mesh_handler.mesh.vert.size()<<endl;
            //cout<<"map.size()"<<map.size()<<endl;
        }

    public:
        void compute_vertexes_per_image(map<int, map<long long, vector<vcg::Point2f>>>& outter_map, const HandleMesh& mesh_handler, const cv::Mat& depthImage, const Eigen::Matrix4d& extrinsic, const Eigen::Matrix3d& intrinsic, long long timestamp){
            
            Eigen::Matrix4d extrinsicInverse = extrinsic.inverse();
            for (int vIdx = 0; vIdx < mesh_handler.mesh.vert.size(); ++vIdx) {
                map<long long, vector<vcg::Point2f>> inner_map;
                vcg::Point3f vertex = mesh_handler.mesh.vert[vIdx].P();
                Eigen::Vector4d vertexHomogeneous(vertex[0], vertex[1], vertex[2], 1.0);
                Eigen::Vector4d camCoords = extrinsicInverse * vertexHomogeneous;
                
                if (camCoords[2] <= 0) continue; // Vertex is behind the camera

                Eigen::Vector3d imageCoords = intrinsic * camCoords.head<3>();
                cv::Point2f pixel(imageCoords[0] / imageCoords[2], imageCoords[1] / imageCoords[2]);

                if (pixel.x >= 0 && pixel.x < depthImage.cols && pixel.y >= 0 && pixel.y < depthImage.rows) {
                    vcg::Point2f value((int)pixel.x,(int)pixel.y);
                    add_value_to_map(inner_map, timestamp, value);
                }
                if(inner_map.size()>0)
                    add_value_to_map(outter_map, vIdx, inner_map);
            }

            //print_map(outter_map);
            //cout<<"mesh_handler.mesh.vert.size() "<<mesh_handler.mesh.vert.size()<<endl;
            //cout<<"map.size()"<<outter_map.size()<<endl;
        }

    public:
        void compute_vertexes_per_image(const HandleMesh& mesh_handler, const cv::Mat& depthImage, const Eigen::Matrix4d& extrinsic, const Eigen::Matrix3d& intrinsic, long long timestamp, string save_path){
            map<int, vector<vcg::Point2f>> map;
            Eigen::Matrix4d extrinsicInverse = extrinsic.inverse();
            for (int vIdx = 0; vIdx < mesh_handler.mesh.vert.size(); ++vIdx) {
                vcg::Point3f vertex = mesh_handler.mesh.vert[vIdx].P();
                Eigen::Vector4d vertexHomogeneous(vertex[0], vertex[1], vertex[2], 1.0);
                Eigen::Vector4d camCoords = extrinsicInverse * vertexHomogeneous;
                
                if (camCoords[2] <= 0) continue; // Vertex is behind the camera

                Eigen::Vector3d imageCoords = intrinsic * camCoords.head<3>();
                cv::Point2f pixel(imageCoords[0] / imageCoords[2], imageCoords[1] / imageCoords[2]);

                if (pixel.x >= 0 && pixel.x < depthImage.cols && pixel.y >= 0 && pixel.y < depthImage.rows) {
                    vcg::Point2f value((int)pixel.x,(int)pixel.y);
                    add_value_to_map(map, vIdx, value);
                }
                
            }
            //cout<<"here map.size() "<<map.size()<<endl;
            if (map.size()>0)
                mapToJson(map, save_path, to_string(timestamp));
        }

    public:
        void get_vetex_to_pixel_dict(string path_to_pv, string path_to_depth_folder, string save_path){
            
            Project_point projector = Project_point(n_threads);
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, n_threads, verbose);
            //
            path_to_pv = path_to_dataset+path_to_pv;
            auto tuple_intrinsics = projector.extract_intrinsics(path_to_pv);
            float done_images = 1;
            //map<int, map<long long, vector<vcg::Point2f>>> map;

            omp_set_num_threads(n_threads);
            #pragma omp parallel for ordered
            for (int i = 0; i < tuple_intrinsics.size(); i+=1){
                //tuple_intrinsics.size()
                Eigen::Matrix3d intrinsic = std::get<1>(tuple_intrinsics[i]);
                Eigen::Matrix4d extrinsic = std::get<2>(tuple_intrinsics[i]);
                
                long long timestamp = std::get<0>(tuple_intrinsics[i]);
                string path_to_depth = path_to_dataset+path_to_depth_folder+"/"+to_string(timestamp)+"_depth.png";
                
                #pragma omp critical
                compute_vertexes_per_image(mesh_handler, loadDepthImage(path_to_depth), extrinsic, intrinsic, timestamp, path_to_dataset+save_path);
                //compute_vertexes_per_image(map, mesh_handler, loadDepthImage(path_to_depth), extrinsic, intrinsic, timestamp);

                #pragma omp critical
                std::cout<<"processed "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;
                done_images++;
            }
            cout<<""<<endl;
            //cout<<"map.size() "<<map.size()<<endl;
        }

    

};