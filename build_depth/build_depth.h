#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include<omp.h>
#include <chrono>
#include <iomanip> //for precison of float in percentage print (not stricltly necessary)

#include <time.h>


#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/affine.hpp>
#include <Eigen/Dense> //#include <Eigen/Dense> this must be included before #include <opencv2/core/eigen.hpp> 
#include <opencv2/core/eigen.hpp>



//imports for mesh handling
#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/update/color.h>


//import export
#include<wrap/io_trimesh/import.h>
#include<wrap/io_trimesh/export.h>

#include <vcg/math/gen_normal.h>
#include <vcg/complex/allocate.h>

//vcgLibForEmbree
#include<wrap/embree/EmbreeAdaptor.h>


using namespace std;
using namespace vcg;
using namespace std::chrono;


class Project_point
{
    private: int n_threads = 1;
    private: bool verbose = false;

    public:
        Project_point(int threads = 1, bool isVerbose = false)
        {
            n_threads = threads;
            verbose = isVerbose;
        }

    public:
        std::vector<cv::Point2d> Project(const std::vector<cv::Point3d> &points, const cv::Mat &intrinsic, const cv::Mat &distortion)
        {
            std::vector<cv::Point2d> result;
            if (!points.empty())
            {
                cv::projectPoints(points, cv::Mat(3, 1, CV_64F, cv::Scalar(0.)), cv::Mat(3, 1, CV_64F, cv::Scalar(0.)), intrinsic, distortion, result);
            }
            return result;
        }

    public:
        void Unproject(vcg::Point3f& result, const vcg::Point2f &point, const double &Z, const Eigen::Matrix3d& intrinsic)
        {
            double f_x = intrinsic(0,0);
            double f_y = intrinsic(1,1);
            double c_x = intrinsic(0,2);
            double c_y = intrinsic(1,2);
           
            //cout << "fx " << f_x << " fy " << f_y << " cx " << c_x << " cy " << c_y << endl;

            result[0] = ((point[0]-c_x)/f_x) * Z;
            result[1] = ((point[1]-c_y)/f_y) * Z;
            result[2] = Z; 
        }

    
    /*
        @param const std::string &csv_path the path to the *_pv.txt generated from the hololense: HoloLens2ForCV/Samples/StreamRecorder/StreamRecorderConverter/process_all.py
        @description This method returns a vector of tuples composed by the:
            - long long: frame_timestamps
            - double: focal_length_x
            - double: focal_length_y
            - Eigen::Matrix4d: pv2world_transforms
            - double: intrinsics_ox  
            - double: intrinsics_oy
            - int: intrinsics_width
            - int: intrinsics_height
        @returns The vectors of tuples of the values extracted from the *_pv.txt

    */
    public:
        std::vector<std::tuple<long long, Eigen::Matrix4d, Eigen::Matrix3d, int, int>> load_pv_data(const std::string &csv_path)
        {

            std::vector<std::tuple<long long, Eigen::Matrix4d, Eigen::Matrix3d, int, int>> pv_data;
            std::ifstream file(csv_path);
            if (!file.is_open())
            {
                throw std::runtime_error("Unable to open file: " + csv_path);
            }

            std::string line;
            std::getline(file, line);

            double intrinsics_ox, intrinsics_oy;
            int intrinsics_width, intrinsics_height;
            std::sscanf(line.c_str(), "%lf,%lf,%d,%d", &intrinsics_ox, &intrinsics_oy, &intrinsics_width, &intrinsics_height);

            long long frame_timestamps;
            double focal_length_x, focal_length_y;

            while (std::getline(file, line))
            {
                std::istringstream ss(line);
                std::string token;
                Eigen::Matrix4d pv2world_transforms = Eigen::Matrix4d::Zero();

                std::getline(ss, token, ',');
                frame_timestamps = std::stoll(token);
                //cout << "frame_timestamps " << endl;

                std::getline(ss, token, ',');
                focal_length_x = std::stod(token);
                //cout << "focal_length_x " << focal_length_x << endl;

                std::getline(ss, token, ',');
                focal_length_y = std::stod(token);
                //cout << "focal_length_y " << focal_length_y << endl;

                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        std::getline(ss, token, ',');
                        pv2world_transforms(i,j) = std::stod(token);
                        //cout << "pv2world_transforms(i,j) " << i << " " << j << " " << pv2world_transforms(i,j)<< endl;
                    }
                }

                Eigen::Matrix3d K = Eigen::Matrix3d::Zero(); //(3, std::vector<double>(3, 0.0));
                K(0,0) = focal_length_x;
                K(1,1) = focal_length_y;
                K(0,2) = intrinsics_ox;
                K(1,2) = intrinsics_oy;
                K(2,2) = 1.0;
                

                pv_data.push_back(std::make_tuple(frame_timestamps, pv2world_transforms, K, intrinsics_width, intrinsics_height));                
            }

            return pv_data; 
        }

        /*
            @param path_to_folder: the path to the folder containing the data extracted using the the hololense: HoloLens2ForCV/Samples/StreamRecorder/StreamRecorderConverter/process_all.py
            @description given the path to the folder, this method checks for the existence of the *_pv.txt file and calls the load_pv_data method.
                Than it extracs the intrinsics for each image.
            @returns it returns a vector of tuple where each tuple is composed by:
                - long long frame_timestamp 
                - Eigen::Matrix3d K, which are the intrinsics of the frame
                - Eigen::Matrix4d pv2world_transforms
        */
    public:
        std::vector<std::tuple<long long, Eigen::Matrix3d, Eigen::Matrix4d>> extract_intrinsics(const std::string &path_to_folder = "./RAW/dump_cnr_c60/", const std::string &path_to_pv = "./resources/dataset/dump_cnr_c60/2023-12-12-105500_pv.txt")
        {
            std::filesystem::path folder(path_to_folder);
            if (!std::filesystem::exists(folder))
            {
                throw std::runtime_error(string(folder) + " does not exist");
            }

            std::filesystem::path pv_info_path;
            for (const auto &entry : std::filesystem::directory_iterator(string(folder)))
            {
                if (entry.path().extension() == ".txt" && entry.path().stem().stem().string().find("pv") != std::string::npos)
                {
                    pv_info_path = entry.path();
                    break;
                }
            }

            if (pv_info_path.empty()){
                throw std::runtime_error("no *pv.txt file found in path " + string(folder) + " check path. ");
            }
            
            std::filesystem::path pv_folder = folder / "PV";
            if (!std::filesystem::exists(pv_folder) || !std::filesystem::is_directory(pv_folder))
            {
                throw std::runtime_error("PV folder does not exist or is not a directory");
            }

            std::vector<std::tuple<long long, Eigen::Matrix4d, Eigen::Matrix3d, int, int>> pv_data = load_pv_data(path_to_pv);            
            std::vector<std::tuple<long long, Eigen::Matrix3d, Eigen::Matrix4d>> intrinsic_per_image;

            for (int pv_id = 0; pv_id < pv_data.size(); pv_id++)
            {
                auto [frame_timestamp, pv2world_transforms, K, width, height] = pv_data[pv_id]; 
                intrinsic_per_image.push_back(std::make_tuple(frame_timestamp, K, pv2world_transforms));
            }

            return intrinsic_per_image;
        }

    Eigen::Matrix4d read_fixed_extrinsics(const std::string &file_path = "./RAW/dump_cnr_c60/"){
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open the file." << std::endl;
        }

        // Read the values from the file
        std::vector<double> values;
        std::string value;
        while (std::getline(file, value, ',')) {
            values.push_back(std::stod(value));
        }

        // Close the file
        file.close();

        // Check if we have exactly 16 values
        if (values.size() != 16) {
            std::cerr << "The file does not contain exactly 16 values." << std::endl;
        }

        // Create a 4x4 Eigen matrix and fill it with the values
        Eigen::Matrix4d fixed_extrinsic;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                fixed_extrinsic(i, j) = values[i * 4 + j];
            }
        }

        return fixed_extrinsic;

    }

    /*
        @param Eigen::Matrix3d &intrinsic: the 3x3 image intrinsics matrix
        @param int image_height
        @param int image_width

        @description 
            This method calculates for each pixel the direction of the ray using the unproject function.
            The values of the point for the directions is set to pixel coords+0.5 so that it starts from the center of the pixel.
            The Z had to be inferred and was choosen to be set to -1 after some experiments (you might need to change it based on the way the 3d model was reconstructed)
            The values of the origin are set to x=0, y=0, z=0
            
            The directions are returned in image(local) scale.

        @returns 
            - std::vector<std::vector<vcg::Point3f>> ray_directions, a matrix of vcg::Point3f that represents directions in image scale
            - std::vector<std::vector<vcg::Point3f>> ray_origins, a matrix of vcg::Point3f that represents origins in image scale

    */
    public:
        std::tuple<std::vector<std::vector<vcg::Point3f>>, std::vector<std::vector<vcg::Point3f>>> ray_direction_per_image(const Eigen::Matrix3d &intrinsic, const int image_height=1080, const int image_width=1920){
            int rows = image_height;
            int cols = image_width;

            std::vector<std::vector<vcg::Point3f>> ray_directions(rows, std::vector<vcg::Point3f>(cols));
            std::vector<std::vector<vcg::Point3f>> ray_origins(rows, std::vector<vcg::Point3f>(cols, vcg::Point3f(0.0f,0.0f,0.0f)));


            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    Unproject(ray_directions[r][cols-c-1], vcg::Point2f(c+0.5f, r+0.5f), -1, intrinsic);
                    //ray_origins[r][cols-c-1] = vcg::Point3f(0.0f,0.0f,0.0f);
                }

            } 

            return std::make_tuple(ray_origins, ray_directions);
        }

    /*
        @param Eigen::Matrix4d &extrinsic: the 4x4 image extrinsics matrix
        @param std::vector<std::vector<vcg::Point3f>> &image_directions: the matrix of the coordinates to be transformed from local scale to world space

        @description 
            Given a set of directions and the 4x4 matrix representing pv2world for each image.
            This method calls coord_to_mesh_space() and returns the matrix of transformed 3d coords.

        @returns std::vector<std::vector<vcg::Point3f>> world_space_directions: the matrix of world space 3d coords

    */
    public:
        void image_to_mesh_space(std::vector<std::vector<vcg::Point3f>>& world_space_directions, const Eigen::Matrix4d &extrinsic, const std::vector<std::vector<vcg::Point3f>> &image_directions) {
            
            int rows = image_directions.size();
            int cols = image_directions[0].size();

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    coord_to_mesh_space(world_space_directions[r][c], extrinsic, image_directions[r][c]);
                }
            }
    }

    /*
        @param Eigen::Matrix4d &extrinsic: the 4x4 image extrinsics matrix
        @param vcg::Point3f& coords: coordinates to be transformed from local scale to world space

        @description 
            Given a set of coords and the 4x4 matrix representing pv2world for each image.
            The directions returned are in world space
            To return the value in world space the operations are the following:
                1. the coordnates are made homogeneous by adding a 4th value to the 3d point
                2. the extrinsics matrix is multiplied by the homogeneous coordinate
                3. the resulting value are normalized deviding the value of the first 3 indexes of the vector by the 4th 
                    value of the same vector

        @returns vcg::Point3f worldCoords: the coords in world space

    */
    public:
        void coord_to_mesh_space(vcg::Point3f& worldCoords, const Eigen::Matrix4d &extrinsic, const vcg::Point3f& coords){
            // Formulate camera coordinates as a homogeneous 4x1 vector
            Eigen::Vector4d cameraCoordsHomogeneous(coords[0], coords[1], coords[2], 1.0f);
            //cameraCoordsHomogeneous << coords[0], coords[1], coords[2], 1.0f;
            Eigen::Vector4d worldCoordsHomogeneous = extrinsic * cameraCoordsHomogeneous;

            // Normalize the homogeneous coordinates
            worldCoords[0] = (worldCoordsHomogeneous(0) / worldCoordsHomogeneous(3));
            worldCoords[1] = (worldCoordsHomogeneous(1) / worldCoordsHomogeneous(3));
            worldCoords[2] = (worldCoordsHomogeneous(2) / worldCoordsHomogeneous(3));
        }

    public: 
        void print_frame_pv(long long timestamp, string path_to_pv){
            std::vector<std::tuple<long long, Eigen::Matrix4d, Eigen::Matrix3d, int, int>> pv_data = load_pv_data(path_to_pv);
            print_frame_pv(timestamp, pv_data);
        }
    
    /*
        This method is a support method to print the values of a specific timestamp extracted from the *_pv.txt file 
    */
    public:
        void print_frame_pv(long long timestamp, std::vector<std::tuple<long long, Eigen::Matrix4d, Eigen::Matrix3d, int, int>> &pv_data){
            for (int i = 0; i < pv_data.size(); i++){
                auto [frame_timestamps, pv2world_transforms, K, width, height] = pv_data[i]; 
                
                if (frame_timestamps==timestamp){
                    std::cout << "timestamp " << frame_timestamps << endl;
                    std::cout << "focal x " << K(0,0) << endl;
                    std::cout << "focal y " << K(1,1) << endl;
                    std::cout << "ox " << K(0,2) << endl;
                    std::cout << "oy " << K(1,2) << endl;
                    std::cout << "width " << width << endl;
                    std::cout << "height " << height << endl;                 

                    // Print the vector
                    for (int i = 0; i < pv2world_transforms.rows(); ++i) {
                        std::cout << "[ ";
                        for (int j = 0; j < pv2world_transforms.cols(); ++j) {
                            std::cout <<  pv2world_transforms(i,j) << " ";
                        }
                        std::cout << "]" << std::endl;
                    }   

                    return;
                }
            }

            std::cout<<"the timestamp " << timestamp << " not found" << endl;
        }
};


class HandleMesh{

    class MyVertex; class MyEdge; class MyFace;
    struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>::AsVertexType, vcg::Use<MyEdge>::AsEdgeType, vcg::Use<MyFace>::AsFaceType> {};
    class MyVertex : public vcg::Vertex< MyUsedTypes, vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::BitFlags, vcg::vertex::VFAdj, vcg::vertex::Qualityf, vcg::vertex::Color4b> {};
    class MyFace : public vcg::Face< MyUsedTypes, vcg::face::FFAdj, vcg::face::VFAdj, vcg::face::Normal3f, vcg::face::VertexRef, vcg::face::BitFlags, vcg::face::Color4b, vcg::face::Qualityf> {};
    class MyEdge : public vcg::Edge<MyUsedTypes, vcg::edge::VertexRef, vcg::edge::BitFlags> {};
    class MyMesh : public vcg::tri::TriMesh< std::vector<MyVertex>, std::vector<MyEdge>, std::vector<MyFace>  > {};

    public:
        MyMesh mesh;
        int n_threads = 1;

    private: bool verbose = false;

    public:
        HandleMesh(){} //default empty costructor

    public: 
        HandleMesh(string path_to_ply, int threads, bool isVerbose = false){
            verbose = isVerbose;
            n_threads = threads;

            if (verbose)
                printf("loading mesh from: %s \n", path_to_ply.c_str());
            
            MyMesh loaded_mesh;
            
            int ret = tri::io::ImporterOFF<MyMesh>::Open(loaded_mesh,path_to_ply.c_str());
            if(ret!=tri::io::ImporterOFF<MyMesh>::NoError)
            {
                throw std::runtime_error("Error loading the mesh from: " + path_to_ply);
            }

            if (verbose)
                printf("Mesh loaded correctly. \n");
            vcg::tri::Append<MyMesh,MyMesh>::MeshCopy(mesh,loaded_mesh);
            
        }

    // Destructor
    public:
        ~HandleMesh() {
            if(verbose)
                std::cout << "MeshHandler destructor called." << std::endl;
        }

    /*
        @param std::vector<std::vector<vcg::Point3f>> &origins
        @param std::vector<std::vector<vcg::Point3f>> &directions
        @param bool use_omp: this parameters defines of openmp should be used in the calculation
        @param bool log, if true some debugging informations will be printed
        @description given origins and direction, shoots a ray form origin towards direction and return a tuple composed by 
            - std::vector<std::vector<bool>> where each cell represent if something has been hit from origin to direction
            - std::vector<std::vector<vcg::Point3f>> where each cell represent the coordinates of what has been hit from origin to direction
            - std::vector<std::vector<float>> where each cell represent the distance of what has been hit from origin to direction
            - std::vector<std::vector<int>> where each cell represent the face id that has been hit from origin to direction
        @returns returns the tuple containing all the matrices:
            std::tuple<std::vector<std::vector<bool>>, std::vector<std::vector<vcg::Point3f>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> 
    */
    public:
        std::tuple<std::vector<std::vector<bool>>, std::vector<std::vector<vcg::Point3f>>, std::vector<std::vector<float>>, std::vector<std::vector<int>>> project_rays(std::vector<std::vector<vcg::Point3f>> &origins, std::vector<std::vector<vcg::Point3f>> &directions, bool use_omp = true, bool log = false){
            int rows = directions.size();
            int cols = directions[0].size();
            
            //std::cout<<"project_rays in rows "<<rows<<" cols "<<cols<<endl;

            std::vector<std::vector<bool>> hit_something_mat(rows, std::vector<bool>(cols));
            std::vector<std::vector<vcg::Point3f>> hit_coords_mat(rows, std::vector<vcg::Point3f>(cols));
            std::vector<std::vector<float>> hit_distances_mat(rows, std::vector<float>(cols));
            std::vector<std::vector<int>> hit_face_id_mat(rows, std::vector<int>(cols));
            
            int total_iterations = rows;
            int progress = 0;
            float scale = 100.0f;
            EmbreeAdaptor<MyMesh> adaptor = EmbreeAdaptor<MyMesh>(mesh);

            for(int r = 0; r < rows; r++){
                progress++;
                for(int c = 0; c < cols; c++){
                    vcg::Point3f origin = origins[r][c];

                    Point3f direction = directions[r][c];
                    Point3f scaledDirection(origin[0] + ((direction[0]-origin[0])*scale),origin[1] + ((direction[1]-origin[1])*scale), origin[2] + ((direction[2]-origin[2])*scale));

                    direction = scaledDirection;

                    direction.Normalize();
                    float tnear = 0.00001f; //std::sqrt(std::pow(direction[0] - origin[0], 2) + std::pow(direction[1] - origin[1], 2) + std::pow(direction[2] - origin[2], 2));
                    auto [hit_something, hit_face_coords, hit_distance, id] = adaptor.shoot_ray(origin, direction, tnear, false);
                
                    hit_something_mat[r][c] = hit_something;
                    if (hit_distance != std::numeric_limits<float>::infinity())
                        hit_distances_mat[r][c] = hit_distance;
                    else
                        hit_distances_mat[r][c] = 0;
                    
                    hit_coords_mat[r][c] = hit_face_coords;
                    
                    hit_face_id_mat[r][c] = id;

                }

                if (progress % 10 == 0 && log)
                    std::cout << "Progress: " << progress << "/"<<rows<<endl;
            }
        
            if (log){
                std::cout << "Progress: 100%"<<endl;
                std::cout << "size of hit_something_mat " << hit_something_mat.size() << "x" << hit_something_mat[0].size() << endl;
                std::cout << "size of hit_coords_mat " << hit_coords_mat.size() << "x" << hit_coords_mat[0].size() << endl;
                std::cout << "size of hit_distances_mat " << hit_distances_mat.size() << "x" << hit_distances_mat[0].size() << endl;
                std::cout << "size of hit_face_id_mat " << hit_face_id_mat.size() << "x" << hit_face_id_mat[0].size() << endl;
            }
            
            //I make sure to release the resources once the ray shooting has finished 
            adaptor.release_global_resources();
            return std::make_tuple(hit_something_mat, hit_coords_mat, hit_distances_mat, hit_face_id_mat);
        }    

    public:
        std::tuple<bool, Point3f, float, int> project_rays(vcg::Point3f origin, vcg::Point3f direction){

            EmbreeAdaptor<MyMesh> adaptor = EmbreeAdaptor<MyMesh>(mesh);
            float scale = 100.0f;
            Point3f scaledDirection(origin[0] + ((direction[0]-origin[0])*scale),origin[1] + ((direction[1]-origin[1])*scale), origin[2] + ((direction[2]-origin[2])*scale));
            direction = scaledDirection;
            direction.Normalize();
            
            float tnear = 0.00001f; //std::sqrt(std::pow(direction[0] - origin[0], 2) + std::pow(direction[1] - origin[1], 2) + std::pow(direction[2] - origin[2], 2));
            //print_point3f(origin,"origin");
            //print_point3f(direction, "direction");
            return adaptor.shoot_ray(origin, direction, tnear, true, true);

        }    

    /*  
        @param std::vector<std::vector<vcg::Point3f>> &origins
        @param std::vector<std::vector<vcg::Point3f>> &directions 
        @description this method adds a vertex for each origin point and each direction and saves it as a .off file
    */
    public:
        void visualize_points_in_mesh(std::vector<std::vector<vcg::Point3f>> &origins, std::vector<std::vector<vcg::Point3f>> &directions, string file_name = "./resources/test_OriginDirections.off"){

            int rows = directions.size();
            int cols = directions[0].size();

            MyMesh mesh2;

            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    vcg::Point3f origin = origins[r][c];
                    vcg::Point3f direction = directions[r][c]; 
                
                    vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, origin);
                    vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, direction);
                }
            }

            tri::io::ExporterPLY<MyMesh>::Save(mesh2, file_name.c_str());

        }

    /*  
        @param vcg::Point3f &origin
        @param std::vector<std::vector<vcg::Point3f>> &directions: it must be a 2x2 matrix
        @param string file_name: the name of the file to be saved 
        @param bool use_edge: if true it saves the vertexes for origin and direction than uses some edges 
            to connect the origin to the direction vertex and than connect the 4 vertices of direction to create a cone.

        @description this method adds a vertex for each origin point and each direction and saves it as a .ply
    */
    public:
        void visualize_points_in_mesh(vcg::Point3f &origin, std::vector<std::vector<vcg::Point3f>> &directions, string file_name = "./resources/test_OriginDirections.off", bool use_edge = false){

            int rows = directions.size();
            int cols = directions[0].size();
            MyMesh mesh2;
            vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, origin);
            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    vcg::Point3f direction = directions[r][c]; 
                
                    vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, direction);
                }
            }

            if(use_edge){
                // Add a face between the two vertices
                vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[1]);
                vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[2]);
                vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[3]);
                vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[4]);

                vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[1], &mesh2.vert[2]);
                vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[1], &mesh2.vert[3]);
                vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[3], &mesh2.vert[4]);
                vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[4], &mesh2.vert[2]);
            }

            int mask = vcg::tri::io::Mask::IOM_VERTCOORD;
            mask |= vcg::tri::io::Mask::IOM_EDGEINDEX;
            tri::io::ExporterPLY<MyMesh>::Save(mesh2, file_name.c_str(), mask);
        }
    
    public:
        void visualize_points_in_mesh(vcg::Point3f &origin, std::vector<std::vector<vcg::Point3f>> &directions, string file_name, bool use_edge, float scale, bool all_edges = false){

            int rows = directions.size();
            int cols = directions[0].size();
            MyMesh mesh2;
            vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, origin);

            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    vcg::Point3f direction = directions[r][c];
                    /*              
                    Point3D newCoordinate = new Point3D { 
                                        A.X + ((B.X - A.X) * distanceToAdjust),
                                        A.Y + ((B.Y - A.Y) * distanceToAdjust),
                                        A.Z + ((B.Z - A.Z) * distanceToAdjust)
                                    }
                    */
                    Point3f scaledDirection(origin[0] + ((direction[0]-origin[0])*scale),origin[1] + ((direction[1]-origin[1])*scale), origin[2] + ((direction[2]-origin[2])*scale));
                    vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, scaledDirection);
                }
            }

            if(use_edge){
                // Add a face between the two vertices
                if(all_edges){
                    for(int i = 1; i < mesh2.vert.size(); i++){
                        vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[i]);
                    }
                }
                else{

                    vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[1]);
                    vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[cols+1]);
                    vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[mesh2.vert.size()-1-cols]);
                    vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[mesh2.vert.size()-1]);

                    vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[1], &mesh2.vert[cols+1]);
                    vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[1], &mesh2.vert[mesh2.vert.size()-1-cols]);
                    vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[mesh2.vert.size()-1-cols], &mesh2.vert[mesh2.vert.size()-1]);
                    vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[mesh2.vert.size()-1], &mesh2.vert[cols+1]);
                }
                
            }

            int mask = vcg::tri::io::Mask::IOM_VERTCOORD;
            mask |= vcg::tri::io::Mask::IOM_EDGEINDEX;
            tri::io::ExporterPLY<MyMesh>::Save(mesh2, file_name.c_str(), mask);
        }

    public:
        void visualize_points_in_mesh(vcg::Point3f &origin, vcg::Point3f& direction, string file_name, bool use_edge, float scale){

            MyMesh mesh2;
            vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, origin);

            Point3f scaledDirection(origin[0] + ((direction[0]-origin[0])*scale),origin[1] + ((direction[1]-origin[1])*scale), origin[2] + ((direction[2]-origin[2])*scale));
            vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, scaledDirection);


            if(use_edge){
                // Add a face between the two vertices
                vcg::tri::Allocator<MyMesh>::AddEdge(mesh2, &mesh2.vert[0], &mesh2.vert[1]);                
            }

            int mask = vcg::tri::io::Mask::IOM_VERTCOORD;
            mask |= vcg::tri::io::Mask::IOM_EDGEINDEX;
            tri::io::ExporterPLY<MyMesh>::Save(mesh2, file_name.c_str(), mask);
        }


    public:
        void visualize_points_in_mesh(vcg::Point3f &origin, std::vector<vcg::Point3f> &directions, string file_name = "./resources/test_OriginDirections.off"){

            MyMesh mesh2;
            vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, origin);
            for(int r = 0; r < directions.size(); r++){
                vcg::Point3f direction = directions[r];                 
                vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, direction);
            }

            int mask = vcg::tri::io::Mask::IOM_VERTCOORD;
            mask |= vcg::tri::io::Mask::IOM_EDGEINDEX;
            tri::io::ExporterPLY<MyMesh>::Save(mesh2, file_name.c_str(), mask);
        }

    /*
        @param string outputMeshPath,  path+name+extention of the mesh to save
        @param int mask, the mask for the mesh to save

        @description this is a wrapper for savig meshes in vcglib
    */
    public:
        void save_mesh(string outputMeshPath, int mask = -1){
            if (mask == -1){
                if(tri::io::Exporter<MyMesh>::Save(mesh, outputMeshPath.c_str()) != 0) {
                    cerr << "Error saving mesh: " << outputMeshPath << endl;
                }
                else
                    cout << "Mesh saved correctly at: " << outputMeshPath << endl;
            }
            else{
                if(tri::io::Exporter<MyMesh>::Save(mesh, outputMeshPath.c_str(), mask) != 0) {
                    cerr << "Error saving mesh: " << outputMeshPath << endl;
                }
                else
                    cout << "Mesh saved correctly at: " << outputMeshPath << endl;
            }
        }

    /*
        @param Eigen::MatrixXf &mat
        @param string &filename the name to give to the file
        @description The saveMatAsCSV are a series of support methods to save the Eigen::MatrixXf in a comma separated value file
    */
    public:
        void saveMatAsCSV(const Eigen::MatrixXf &mat, const string &filename) {
            ofstream file(filename);

            if (!file.is_open()) {
                cerr << "Error: Unable to open file for writing! in: saveMatAsCSV Eigen::MatrixXf" << endl;
                return;
            }

            for (int i = 0; i < mat.rows(); i++) {
                for (int j = 0; j < mat.cols(); j++) {
                    file << mat(i, j); 
                    if (j < mat.cols() - 1) {
                        file << ","; // Add comma except for the last element in each row
                    }
                }
                file << endl; // End line after each row
            }

            file.close();
        }

    /*
        @param std::vector<std::vector<int>> &mat
        @param string &filename the name to give to the file
        @description The saveMatAsCSV are a series of support methods to save the std::vector<std::vector<int>> in a comma separated value file
    */
    public:
        void saveMatAsCSV(const std::vector<std::vector<int>> &mat, const string &filename) {
            ofstream file(filename);

            if (!file.is_open()) {
                cerr << "Error: Unable to open file for writing! in: saveMatAsCSV vector<vector<int>>" << endl;
                return;
            }

            int cols = mat[0].size();
            int rows = mat.size();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    file << mat[r][c]; 
                    if (c < cols - 1) {
                        file << ","; // Add comma except for the last element in each row
                    }
                }
                file << endl; // End line after each row
            }

            file.close();
        }

    /*
        @param std::vector<std::vector<float>> &mat
        @param string &filename the name to give to the file
        @description The saveMatAsCSV are a series of support methods to save the std::vector<std::vector<float>> in a comma separated value file
    */
    public:
        void saveMatAsCSV(const std::vector<std::vector<float>> &mat, const string &filename) {
            ofstream file(filename);

            if (!file.is_open()) {
                cerr << "Error: Unable to open file for writing! in: saveMatAsCSV vector<vector<float>>" << endl;
                return;
            }

            int cols = mat[0].size();
            int rows = mat.size();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    file << mat[r][c]; 
                    if (c < cols - 1) {
                        file << ","; // Add comma except for the last element in each row
                    }
                }
                file << endl; // End line after each row
            }

            file.close();
        }
    
    /*
        @param std::vector<std::vector<bool>> &mat
        @param string &filename the name to give to the file
        @description The saveMatAsCSV are a series of support methods to save the std::vector<std::vector<bool>> in a comma separated value file
    */
    public:
        void saveMatAsCSV(const std::vector<std::vector<bool>> &mat, const string &filename) {
            ofstream file(filename);

            if (!file.is_open()) {
                cerr << "Error: Unable to open CSV file for writing! in: saveMatAsCSV vector<vector<bool>> " << endl;
                return;
            }

            int cols = mat[0].size();
            int rows = mat.size();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    file << mat[r][c]; 
                    if (c < cols - 1) {
                        file << ","; // Add comma except for the last element in each row
                    }
                }
                file << endl; // End line after each row
            }


            file.close();
        }

    /*
        @param std::vector<std::vector<vcg::Point3f>> &mat
        @param string &filename the name to give to the file
        @description The saveMatAsCSV are a series of support methods to save the std::vector<std::vector<vcg::Point3f>> in a comma separated value file
    */
    public:
        void saveMatAsCSV(const std::vector<std::vector<vcg::Point3f>> &mat, const string &filename) {
            ofstream file(filename);

            if (!file.is_open()) {
                cerr << "Error: Unable to open file for writing! in: saveMatAsCSV vector<vector<vcg::Point3f>>" << endl;
                return;
            }

            int cols = mat[0].size();
            int rows = mat.size();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {  
                    vcg::Point3f coords = mat[r][c];    
                    string element = "("+std::to_string(coords[0])+" "+std::to_string(coords[1])+" "+std::to_string(coords[2])+")";
                    file << element;
                    if (c < cols - 1) {
                        file << ","; // Add comma except for the last element in each row
                    }
                }
                file << endl; // End line after each row
            }

            file.close();
        }

    /*
        @param std::vector<std::vector<float>> &floatMat
        @param string &filename the name to give to the file
        @description given a matrix of floats normalize it to rewrite it as a 16-bit grayscale image
    */    
    public:
        void saveFloatMatAsGrayscaleImage(const std::vector<std::vector<float>> &floatMat, const std::string& filename) {
            int rows = floatMat.size();
            int cols = floatMat[0].size();
            cv::Mat matrix(rows, cols, CV_32FC1);//CV_16UC1
            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    matrix.at<float>(r, c) = floatMat[r][c]; //* 5000; 
                }
            }

            cv::imwrite(filename, matrix);
        }

    public:
        void saveFloatMatAsGrayscaleImage(const std::vector<std::vector<float>> &floatMat, const std::string& filename, float depth_scale) {
            int rows = floatMat.size();
            int cols = floatMat[0].size();
            cv::Mat matrix(rows, cols, CV_16UC1);//CV_16UC1
            for(int r = 0; r < rows; r++){
                for(int c = 0; c < cols; c++){
                    matrix.at<ushort>(r, c) = floatMat[r][c] * depth_scale; 
                }
            }

            cv::imwrite(filename, matrix);
        }

    
    public:
        void select_vertex_from_map(map<int, vector<vcg::Point2f>>& map, string save_path, string filename){
            // Check if the directory already exists
            if (!filesystem::exists(save_path)) {
                // Create the directory
                if (filesystem::create_directory(save_path)) {
                    std::cout << "Directory created successfully: "<< save_path << std::endl;
                } else {
                    std::cerr << "Error: Failed to create directory: " << save_path << std::endl;
                }
            }

            MyMesh mesh2;

            for(auto it = map.cbegin(); it != map.cend(); ++it)
            {
                int key = it->first;
                vcg::Point3f coords = mesh.vert[key].P();
                vcg::tri::Allocator<MyMesh>::AddVertex(mesh2, coords);
            }

            int mask = vcg::tri::io::Mask::IOM_VERTCOORD;
            mask |= vcg::tri::io::Mask::IOM_EDGEINDEX;
            tri::io::ExporterPLY<MyMesh>::Save(mesh2, (save_path+"/"+filename).c_str(), mask);
        }


    public:
        void print_point3f(vcg::Point3f& point, string message="", bool vertical = false){
            if(vertical)
                cout<<message<<"\nx: " << point[0] << "\ny: " << point[1] << "\nz: " << point[2] << endl;
            else
                cout<<message<<" x: " << point[0] << " y: " << point[1] << " z: " << point[2] << endl;

        } 

    public:
        void print_point2f(vcg::Point2f& point, string message="", bool vertical = false){
            if(vertical)
                cout<<message<<"\nx: " << point[0] << "\ny: " << point[1] << endl;
            else
                cout<<message<<" x: " << point[0] << " y: " << point[1] << endl;

        } 

};