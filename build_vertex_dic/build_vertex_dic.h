#include "../build_depth/build_depth.h"
#include <map>
#include <utility>      // std::pair, std::make_pair
#include <nlohmann/json.hpp>

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
        map<int, vector<vcg::Point2f>> compute_vertexes_per_image(const HandleMesh& mesh_handler, const cv::Mat& depthImage, const Eigen::Matrix4d& extrinsic, const Eigen::Matrix3d& intrinsic, long long timestamp, string save_path, float depthScale = 5000, bool save_json = true, bool draw_bb = false){
            map<int, vector<vcg::Point2f>> map;           
            std::vector<int> valid_vertex = get_possible_vertex_id(mesh_handler, depthImage, extrinsic, intrinsic);
            //cout << "get_possible_vertex_id: " << valid_vertex.size() <<" out of "<< mesh_handler.mesh.vert.size() <<endl;
            
            Project_point point_projector = Project_point();
            //Eigen::Matrix4d extrinsicInverse = extrinsic.inverse();

            
            std::vector<std::vector<vcg::Point3f>> ray_direction_ws(depthImage.rows, std::vector<vcg::Point3f>(depthImage.cols));
            for (int y = 0; y < depthImage.rows; y++) {
                //cout << y << "\r" << flush;
                for (int x = 0; x < depthImage.cols; x++) { //depthImage.cols
                    //cout << y << ","<<x<<"\r"<<flush; 
                    // Read depth value from depth map (assuming single-channel float)
                    uint16_t depthValue = depthImage.at<uint16_t>(y, x);
                    float depth = depthValue * (1/depthScale);

                    if (depth > 0) { // Valid depth value
                        // Compute 3D point in camera coordinates
                        vcg::Point2f pixel(x,y);
                        vcg::Point3f unprojectd_pixel = point_projector.Unproject(pixel, -depth, intrinsic);
                        vcg::Point3f worldPoint = point_projector.coord_to_mesh_space(extrinsic, unprojectd_pixel);
                        
                        ray_direction_ws[y][x] = worldPoint; 

                        // Iterate over point cloud vertices to find the closest vertex
                        float minDist = std::numeric_limits<float>::max();
                        int closestVertexIdx = -1;
                        
                        for (int vIdx = 0; vIdx < valid_vertex.size(); ++vIdx) {
                            //cout << " checking vertex "<< vIdx << "         \r"<< std::flush;
                            vcg::Point3f vertex = mesh_handler.mesh.vert[valid_vertex[vIdx]].P();
                            // Calculate distance between worldPoint and vertexPos
                            float distance = (worldPoint - vertex).Norm(); // Euclidean distance
                            //cout << "distance "<< distance << endl;
                            if (distance < minDist) {
                                minDist = distance;
                                closestVertexIdx = vIdx;

                                if (minDist < 0.01)
                                    break;
                            }                       
                        }
                        
                        //cout << "picked "<<closestVertexIdx << " with distance "<< minDist<<endl;
                        add_value_to_map(map, closestVertexIdx, pixel);   
                    }
                }
                
            }
            
            if(draw_bb){    
                std::vector<std::vector<vcg::Point3f>> direction(2, std::vector<vcg::Point3f>(2));
                direction[0][0] = ray_direction_ws[0][0];
                direction[0][1] = ray_direction_ws[0][depthImage.cols-1];
                direction[1][0] = ray_direction_ws[depthImage.rows-1][0];
                direction[1][1] = ray_direction_ws[depthImage.rows-1][depthImage.cols-1];
                vcg::Point3f origin(0,0,0);
                HandleMesh().visualize_points_in_mesh(origin, direction,path_to_dataset+"cnr_c60/ply_files/BB_"+to_string(timestamp)+"_4.ply", true);
            }    

            //cout<<"here map.size() "<<map.size()<<endl;
            if (map.size()>0 && save_json)
                map_to_json(map, save_path, to_string(timestamp));
            
            return map;
        }

    public:
        std::vector<int> get_possible_vertex_id(const HandleMesh& mesh_handler, const cv::Mat& depthImage, const Eigen::Matrix4d& extrinsic, const Eigen::Matrix3d& intrinsic, float depthScale = 5000, float threshold = 0.1, string save_path = "cnr_c60/ply_files/", bool save_sphere = false){
            
            Project_point point_projector = Project_point();

            std::vector<int> valid_vrtx_id;

            uint16_t depthValue = depthImage.at<uint16_t>(0, 0);
            float depth = depthValue * (1/depthScale);
            vcg::Point3f worldPoint_0;
            if(depth > 0){
                vcg::Point3f unprojectd_pixel = point_projector.Unproject(vcg::Point2f(0,0), -depth, intrinsic);
                worldPoint_0 = point_projector.coord_to_mesh_space(extrinsic, unprojectd_pixel);
            }


            uint16_t depthValue_middle = depthImage.at<uint16_t>(depthImage.rows/2, depthImage.cols/2);
            float depth_middle = depthValue_middle * (1/depthScale);

            vcg::Point3f worldPoint_middle;
            if(depth_middle > 0){
                vcg::Point3f unprojectd_pixel = point_projector.Unproject(vcg::Point2f(depthImage.cols/2, depthImage.rows/2), -depth_middle, intrinsic);
                worldPoint_middle = point_projector.coord_to_mesh_space(extrinsic, unprojectd_pixel);
            }

            float distance_0_middle = (worldPoint_middle - worldPoint_0).Norm();
            float distance_and_threshold = abs(distance_0_middle) + threshold;
            std::vector<vcg::Point3f> directions;

            for (int vIdx = 0; vIdx < mesh_handler.mesh.vert.size(); ++vIdx) {
                vcg::Point3f vertex = mesh_handler.mesh.vert[vIdx].P();
                float distance = (vertex - worldPoint_middle).Norm();

                if(abs(distance_and_threshold) >= abs(distance)){
                    valid_vrtx_id.push_back(vIdx);          
                    if (save_sphere){
                        directions.push_back(vertex);
                    }
              
                }

            }

            if (save_sphere){
                // Check if the directory already exists
                if (!filesystem::exists(path_to_dataset+save_path)) {
                    // Create the directory
                    if (filesystem::create_directory(path_to_dataset+save_path)) {
                        std::cout << "Directory created successfully: "<< path_to_dataset+save_path << std::endl;
                    } else {
                        std::cerr << "Error: Failed to create directory: " << path_to_dataset+save_path << std::endl;
                    }
                }
                HandleMesh().visualize_points_in_mesh(worldPoint_middle, directions, path_to_dataset+save_path+"SPH_4.ply");

            }

            return valid_vrtx_id;
        }

    public:
        auto get_vetex_to_pixel_dict(string path_to_pv, string path_to_depth_folder, string save_path){
            
            Project_point projector = Project_point(1);
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, 1, verbose);
            //
            path_to_pv = path_to_dataset+path_to_pv;
            auto tuple_intrinsics = projector.extract_intrinsics(path_to_pv);
            float done_images = 1;
            map<long long, map<int, vector<vcg::Point2f>>> map;

            omp_set_num_threads(n_threads);
            #pragma omp parallel for 
            for (int i = 0; i < tuple_intrinsics.size(); i+=1){
                Eigen::Matrix3d intrinsic = std::get<1>(tuple_intrinsics[i]);
                Eigen::Matrix4d extrinsic = std::get<2>(tuple_intrinsics[i]);
                
                long long timestamp = std::get<0>(tuple_intrinsics[i]);
                string path_to_depth = path_to_dataset+path_to_depth_folder+"/"+to_string(timestamp)+"_depth.png";
                
                map[timestamp] = compute_vertexes_per_image(mesh_handler, loadDepthImage(path_to_depth), extrinsic, intrinsic, timestamp, path_to_dataset+save_path);

                #pragma omp critical
                std::cout<<"processed "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;
                done_images++;
            }
            cout<<""<<endl;
            //cout<<"map.size() "<<map.size()<<endl;
            return map;
        }
};