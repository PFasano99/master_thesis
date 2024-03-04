#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <fstream>
#include <filesystem>
#include<omp.h>

using namespace std;

class Project_point
{

    private: int n_threads = 1;

    public:
        Project_point(int threads = 4)
        {
            n_threads = threads;
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
        std::vector<cv::Point3d> Unproject(const std::vector<cv::Point2d> &points, const std::vector<double> &Z, const cv::Mat &intrinsic, const cv::Mat &distortion)
        {
            double f_x = intrinsic.at<double>(0, 0);
            double f_y = intrinsic.at<double>(1, 1);
            double c_x = intrinsic.at<double>(0, 2);
            double c_y = intrinsic.at<double>(1, 2);

            // Step 1. Undistort
            std::vector<cv::Point2d> points_undistorted;
            assert(Z.size() == 1 || Z.size() == points.size());
            if (!points.empty())
            {
                cv::undistortPoints(points, points_undistorted, intrinsic,
                                    distortion, cv::noArray(), intrinsic);
            }

            // Step 2. Reproject
            std::vector<cv::Point3d> result;
            result.reserve(points.size());
            for (size_t idx = 0; idx < points_undistorted.size(); ++idx)
            {
                const double z = Z.size() == 1 ? Z[0] : Z[idx];
                result.push_back(
                    cv::Point3d((points_undistorted[idx].x - c_x) / f_x * z,
                                (points_undistorted[idx].y - c_y) / f_y * z, z));
            }
            return result;
        }

    public:
        std::vector<std::tuple<long long, double, double, std::array<std::array<double, 4>, 4>, double, double, int, int>> load_pv_data(const std::string &csv_path)
        {
            std::vector<std::tuple<long long, double, double, std::array<std::array<double, 4>, 4>, double, double, int, int>> pv_data;
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
                std::array<double, 4> transform_row;
                std::array<std::array<double, 4>, 4> pv2world_transforms;

                std::getline(ss, token, ',');
                frame_timestamps = std::stoll(token);

                std::getline(ss, token, ',');
                focal_length_x = std::stod(token);

                std::getline(ss, token, ',');
                focal_length_y = std::stod(token);

                for (int i = 0; i < 4; ++i)
                {
                    for (int j = 0; j < 4; ++j)
                    {
                        std::getline(ss, token, ',');
                        transform_row[j] = std::stod(token);
                    }
                    pv2world_transforms[i] = transform_row;
                }

                pv_data.push_back(std::make_tuple(frame_timestamps, focal_length_x, focal_length_y, pv2world_transforms, intrinsics_ox, intrinsics_oy, intrinsics_width, intrinsics_height));                
            }

            return pv_data; //std::make_tuple(frame_timestamps, focal_lengths, pv2world_transforms, intrinsics_ox, intrinsics_oy, intrinsics_width, intrinsics_height);
        }


    public:
        std::vector<std::tuple<long long, cv::Mat, cv::Mat>> extract_intrinsics(const std::string &path_to_folder = "./RAW/dump_cnr_c60/")
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

            std::vector<std::filesystem::path> pv_timestamps;
            for (const auto &entry : std::filesystem::directory_iterator(pv_folder))
            {
                if (entry.path().extension() == ".png")
                {
                    pv_timestamps.push_back(entry.path().filename().string());
                }
            }

            if (pv_timestamps.empty())
            {
                throw std::runtime_error("No PNG files found in the PV folder");
            }


            std::vector<std::tuple<long long, double, double, std::array<std::array<double, 4>, 4>, double, double, int, int>> pv_data = Project_point().load_pv_data("./resources/dump_cnr_c60/2023-12-12-105500_pv.txt");

            int n_frames = pv_timestamps.size();
            std::vector<std::tuple<long long, cv::Mat, cv::Mat>> intrinsic_per_image;

            // Set the number of threads
            omp_set_num_threads(n_threads);
            #pragma omp parallel for
            for (int pv_id = 0; pv_id < n_frames-1; pv_id++)
            {
                auto [frame_timestamp, focal_length_x, focal_length_y, pv2world_transforms, ox, oy, width, height] = pv_data[pv_id]; 
              
                cv::Mat K = (cv::Mat_<double>(3, 3) << focal_length_x, 0, ox,
                            0, focal_length_y, oy,
                            0, 0, 1);

                cv::Mat P2W(4, 4, CV_64FC1);

                // Copy the values from the array to the cv::Mat
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        P2W.at<double>(i, j) = pv2world_transforms[i][j];
                    }
                }

                
                #pragma omp critical
                intrinsic_per_image.push_back(std::make_tuple(frame_timestamp, K, P2W));
            }

            return intrinsic_per_image;
        }
    


    public:
        std::tuple<cv::Mat, cv::Mat> ray_direction_per_image(const cv::Mat &intrinsic, const cv::Mat &distortion, const cv::Mat &image){
            /*
                This method calculates for each pixel the direction of the ray using the unproject function.
                The orgin of the point is set to pixel coords+0.5 so that it starts from the center of the pixel.
                The directions are returned in image(local) scale
            */
            int image_height = image.size[0];
            int image_width = image.size[1];

            cv::Mat ray_directions(image_height, image_width, CV_64FC3);
            cv::Mat ray_origins(image_height, image_width, CV_64FC3);

            for(int h = 0.5f; h < image_height; h++){
                for(int w = 0.5f; w < image_width; w++){
                    const cv::Point2d point_single(h, w);
                    ray_directions.at<cv::Point3d>(h-0.5f, w-0.5f) = Project_point().Unproject({point_single},{1}, intrinsic, distortion)[0];
                    ray_origins.at<cv::Point3d>(h, w) = cv::Point3d(h,w,0); 
                }
            } 

            return std::make_tuple(ray_origins, ray_directions);
        }


    public:
        cv::Mat image_to_mesh_space(const cv::Mat &intrinsic, const cv::Mat &image_directions) {
            /*
                Given a set of directions and the 4x4 matrix representing pv2world for each image.
                The directions returned are in world space
            */
            //printf("Transforming directions from image space to world space... \n");
            int image_height = image_directions.rows;
            int image_width = image_directions.cols;

            cv::Mat world_space_directions(image_height, image_width, CV_64FC3);
            cv::Mat intrinsic_inverse = intrinsic.inv();

            for (int h = 0; h < image_height; h++) {
                for (int w = 0; w < image_width; w++) {
                    cv::Point3d directions = image_directions.at<cv::Point3d>(h, w);

                    // Formulate camera coordinates as a homogeneous 4x1 vector
                    cv::Mat cameraCoordsHomogeneous = (cv::Mat_<double>(4, 1) <<
                        directions.x, directions.y, directions.z, 1.0);

                    // Multiply the inverse intrinsics matrix by the camera coordinates
                    cv::Mat worldCoordsHomogeneous = intrinsic_inverse * cameraCoordsHomogeneous;

                    // Normalize the homogeneous coordinates
                    cv::Point3d worldCoords(
                        worldCoordsHomogeneous.at<double>(0, 0) / worldCoordsHomogeneous.at<double>(3, 0),
                        worldCoordsHomogeneous.at<double>(1, 0) / worldCoordsHomogeneous.at<double>(3, 0),
                        worldCoordsHomogeneous.at<double>(2, 0) / worldCoordsHomogeneous.at<double>(3, 0)
                    );

                    world_space_directions.at<cv::Point3d>(h, w) = worldCoords;
                }
            }

            return world_space_directions;
    }



    public:
        void print_directions_matrix(cv::Mat &ray_directions){
            /*
                Given a cv::Mat prints it cell by cell, line by line
            */
            int height = ray_directions.size[0];
            int width = ray_directions.size[1];

            for(int h = 0; h < height; h++){
                for (int w = 0; w < width; w++){
                    cv::Vec3d direction = ray_directions.at<cv::Vec3d>(h, w);
                    printf("directoin at cell %d,%d is (%lf, %lf, %lf) \n", h,w,direction[0],direction[1],direction[2]);
                }
            }

        }

    public: 
        void print_frame_pv(long long timestamp, string path_to_pv){
            std::vector<std::tuple<long long, double, double, std::array<std::array<double, 4>, 4>, double, double, int, int>> pv_data = Project_point().load_pv_data(path_to_pv);
            print_frame_pv(timestamp, pv_data);
        }
    
    public:
        void print_frame_pv(long long timestamp, std::vector<std::tuple<long long, double, double, std::array<std::array<double, 4>, 4>, double, double, int, int>> &pv_data){
            for (int i = 0; i < pv_data.size(); i++){
                auto [frame_timestamps, focal_length_x, focal_length_y, pv2world_transforms, ox, oy, width, height] = pv_data[i]; 
                
                if (frame_timestamps==timestamp){
                    std::cout << "timestamp " << frame_timestamps << endl;
                    std::cout << "focal x " << focal_length_x << endl;
                    std::cout << "focal y " << focal_length_y << endl;
                    std::cout << "ox " << ox << endl;
                    std::cout << "oy " << oy << endl;
                    std::cout << "width " << width << endl;
                    std::cout << "height " << height << endl;

                        // Print the vector
                    for (const auto& arr : pv2world_transforms) {
                        std::cout << "[ ";
                        for (const auto& num : arr) {
                            std::cout << num << " ";
                        }
                        std::cout << "]" << std::endl;
                    }   

                    return;
                }
            }

            std::cout<<"the timestamp " << timestamp << " not found" << endl;
        }
};

#include <vcg/complex/complex.h>
#include <vcg/complex/algorithms/create/platonic.h>

//import export
#include<wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export_off.h>
#include <wrap/io_trimesh/import_off.h>

#include <time.h>
#include <vcg/math/gen_normal.h>
#include <vcg/complex/allocate.h>

//vcgLibForEmbree
#include<wrap/embree/EmbreeAdaptor.h>

using namespace vcg; 

class HandleMesh{

    class MyVertex; class MyEdge; class MyFace;
    struct MyUsedTypes : public vcg::UsedTypes<vcg::Use<MyVertex>   ::AsVertexType,
        vcg::Use<MyEdge>     ::AsEdgeType,
        vcg::Use<MyFace>     ::AsFaceType> {};

    class MyVertex : public vcg::Vertex< MyUsedTypes, vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::BitFlags, vcg::vertex::VFAdj, vcg::vertex::Qualityf, vcg::vertex::Color4b> {};
    class MyFace : public vcg::Face< MyUsedTypes, vcg::face::FFAdj, vcg::face::VFAdj, vcg::face::Normal3f, vcg::face::VertexRef, vcg::face::BitFlags, vcg::face::Color4b, vcg::face::Qualityf> {};
    class MyEdge : public vcg::Edge< MyUsedTypes> {};

    class MyMesh : public vcg::tri::TriMesh< std::vector<MyVertex>, std::vector<MyFace>, std::vector<MyEdge>  > {};

    public:
        MyMesh mesh;
        int n_threads = 1;

    public: 
        HandleMesh(string path_to_ply, int threads){

            n_threads = threads;

            printf("loading mesh from: %s \n", path_to_ply.c_str());
            
            MyMesh loaded_mesh;
            
            int ret = tri::io::ImporterOFF<MyMesh>::Open(loaded_mesh,path_to_ply.c_str());
            if(ret!=tri::io::ImporterOFF<MyMesh>::NoError)
            {
                printf("Error loading the mesh from: %s \n", path_to_ply.c_str());
                exit(0);
            }
            printf("Mesh loaded correctly. \n");
            vcg::tri::Append<MyMesh,MyMesh>::MeshCopy(mesh,loaded_mesh);
            
        }


    /*
        @param cv::Mat &origins, a cv::Mat representing the cv::Point3d of the origins
        @param cv::Mat &directions, a cv::Mat representing the cv::Point3d of the directions
        @description given origins and direction, shoots a ray form origin twards direction and return a tuple composed by 
            - cv::mat where each cell represent if something has been hit from origin to direction
            - cv::mat where each cell represent the coordinates of what has been hit from origin to direction
            - cv::mat where each cell represent the distance of what has been hit from origin to direction
            - cv::mat where each cell represent the face id that has been hit from origin to direction
    */
    public:
        std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> project_rays(cv::Mat &origins, cv::Mat &directions, bool use_omp = true, bool log = false){

            int height = directions.size[0];
            int width = directions.size[1];

            cv::Mat hit_something_mat(height, width, CV_32S);
            cv::Mat hit_coords_mat(height, width, CV_64FC3);
            cv::Mat hit_distances_mat(height, width, CV_64F);
            cv::Mat hit_face_id_mat(height, width, CV_64F);

            int total_iterations = height; //height * width;
            int progress = 0;

            EmbreeAdaptor<MyMesh> adaptor = EmbreeAdaptor<MyMesh>(mesh);

            if (use_omp){
                // Set the number of threads
                omp_set_num_threads(n_threads);
                #pragma omp parallel for //collapse(2) schedule(dynamic)
                for(int h = 0; h < height; h++){
                    #pragma omp atomic
                    ++progress;

                    for(int w = 0; w < width; w++){
                        
                        vcg::Point3f origin(static_cast<float>(origins.at<cv::Point3d>(h,w).x), static_cast<float>(origins.at<cv::Point3d>(h,w).y), static_cast<float>(origins.at<cv::Point3d>(h,w).z));
                        vcg::Point3f direction(static_cast<float>(directions.at<cv::Point3d>(h,w).x), static_cast<float>(directions.at<cv::Point3d>(h,w).y), static_cast<float>(origins.at<cv::Point3d>(h,w).z));
                        //cout<<"here d origin x "<< origin[0]<< " y " << origin[1]<< " z " << origin[2] <<endl;
                        //cout<<"here d direction x "<< direction[0]<< " y " << direction[1]<< " z " << direction[2] <<endl;
                        
                        auto [hit_something, hit_face_coords, hit_distance, id ] = adaptor.shoot_ray(origin, direction, false);
                        
                        if(h >= height && w >= width){
                            //adaptor.release_global_resources();
                        }
                        
                        //cout << hit_something <<endl;
                        //cout << hit_face_coords[0] << " " << hit_face_coords[1] << " " << hit_face_coords[2]  <<endl;
                        //cout << hit_distance <<endl;
                        //cout << id <<endl;

                        #pragma opm critical
                        hit_something_mat.at<int>(h, w) = hit_something;
                        #pragma opm critical
                        hit_coords_mat.at<cv::Point3d>(h,w) = cv::Point3d(hit_face_coords[0],hit_face_coords[1],hit_face_coords[2]);
                        #pragma opm critical
                        if (hit_distance != std::numeric_limits<float>::infinity()){
                            hit_distances_mat.at<double>(h,w) = hit_distance;
                        }
                        else
                            hit_distances_mat.at<double>(h,w) = 0;

                        #pragma opm critical
                        hit_face_id_mat.at<int>(h,w) = id;
                                // Calculate progress percentage
                    }
                        int progress_percentage = int((static_cast<double>(progress) / total_iterations) * 100);

                        // Print progress
                        #pragma omp critical
                        if (progress_percentage % 6 == 0 and log)
                            std::cout << "Progress: " << progress_percentage << "%\n";
                }

            }
            else{
                for(int h = 0; h < height; h++){
                    ++progress;

                    for(int w = 0; w < width; w++){
                        
                        vcg::Point3f origin(static_cast<float>(origins.at<cv::Point3d>(h,w).x), static_cast<float>(origins.at<cv::Point3d>(h,w).y), static_cast<float>(origins.at<cv::Point3d>(h,w).z));
                        vcg::Point3f direction(static_cast<float>(directions.at<cv::Point3d>(h,w).x), static_cast<float>(directions.at<cv::Point3d>(h,w).y), static_cast<float>(origins.at<cv::Point3d>(h,w).z));
                        //cout<<"here d origin x "<< origin[0]<< " y " << origin[1]<< " z " << origin[2] <<endl;
                        //cout<<"here d direction x "<< direction[0]<< " y " << direction[1]<< " z " << direction[2] <<endl;
                        
                        auto [hit_something, hit_face_coords, hit_distance, id ] = adaptor.shoot_ray(origin, direction, false);
                        
                        if(h >= height && w >= width){
                            adaptor.release_global_resources();
                        }
                        
                        //cout << hit_something <<endl;
                        //cout << hit_face_coords[0] << " " << hit_face_coords[1] << " " << hit_face_coords[2]  <<endl;
                        //cout << hit_distance <<endl;
                        //cout << id <<endl;

                        hit_something_mat.at<int>(h, w) = hit_something;
                        hit_coords_mat.at<cv::Point3d>(h,w) = cv::Point3d(hit_face_coords[0],hit_face_coords[1],hit_face_coords[2]);
                        if (hit_distance != std::numeric_limits<float>::infinity()){
                            hit_distances_mat.at<double>(h,w) = hit_distance;
                        }
                        else
                            hit_distances_mat.at<double>(h,w) = 0;

                        hit_face_id_mat.at<int>(h,w) = id;
                                // Calculate progress percentage
                    }
                        int progress_percentage = int((static_cast<double>(progress) / total_iterations) * 100);

                        // Print progress
                        if (progress_percentage % 6 == 0 and log)
                            std::cout << "Progress: " << progress_percentage << "%\n";
                }
            }
             
            if (log)
                std::cout << "Progress: 100%"<<endl;

            return std::make_tuple(hit_something_mat, hit_coords_mat, hit_distances_mat, hit_face_id_mat);

        }    

    public:
        std::tuple<bool, Point3f, float, int> project_rays(vcg::Point3f origin, vcg::Point3f direction){

            EmbreeAdaptor<MyMesh> adaptor = EmbreeAdaptor<MyMesh>(mesh);

            auto [hit_something, hit_face_coords, hit_distance, id ] = adaptor.shoot_ray(origin, direction, true);

            return std::make_tuple(hit_something, hit_face_coords, hit_distance, id);

        }    

    /*  
        @param cv::Mat &origins, cv::Mat &directions the cv::Mat composed by 3dpoints to spawn in the scene
        @desciption this method adds a vertex for each origin point and each direction
    */
    public:
        void visualize_points_in_mesh(cv::Mat &origins, cv::Mat &directions){
            int height = directions.size[0];
            int width = directions.size[1];

            for(int h = 0; h < height; h++){
                for(int w = 0; w < width; w++){
                    
                    vcg::Point3f origin(static_cast<float>(origins.at<cv::Point3d>(h,w).x), static_cast<float>(origins.at<cv::Point3d>(h,w).y), static_cast<float>(origins.at<cv::Point3d>(h,w).z));
                    vcg::Point3f direction(static_cast<float>(directions.at<cv::Point3d>(h,w).x), static_cast<float>(directions.at<cv::Point3d>(h,w).y), static_cast<float>(origins.at<cv::Point3d>(h,w).z));
                    //cout<<"here d origin x "<< origin[0]<< " y " << origin[1]<< " z " << origin[2] <<endl;

                    vcg::tri::Allocator<MyMesh>::AddVertex(mesh, origin);
                    vcg::tri::Allocator<MyMesh>::AddVertex(mesh, direction);
                    //newVertexO->P() = origin;

                    //MyVertex *newVertexD = mesh.vert.add();
                    //newVertexD->P() = direction;
                    
                }

             
            }

            tri::io::ExporterOFF<MyMesh>::Save(mesh,"./resources/test_OriginDirections.off");

        }


    /*
        @param cv::Mat &mat the cv::Mat to save
        @param string &filename the name togive to the file
        @Description Given a cv::Mat save its values in a csv format
    */
    public:
        void saveMatAsCSV(const cv::Mat &mat, const string &filename) {
            ofstream file(filename);

            if (!file.is_open()) {
                cerr << "Error: Unable to open file for writing!" << endl;
                return;
            }

            for (int i = 0; i < mat.rows; ++i) {
                for (int j = 0; j < mat.cols; ++j) {
                    file << mat.at<double>(i, j); 
                    if (j < mat.cols - 1) {
                        file << ","; // Add comma except for the last element in each row
                    }
                }
                file << endl; // End line after each row
            }

            file.close();
        }

    public:
        void saveFloatMatAsGrayscaleImage(const cv::Mat& floatMat, const std::string& filename) {
            int height = floatMat.size[0];
            int width = floatMat.size[1];

            cv::Mat zerosMat = cv::Mat::zeros(height, width, CV_8UC1);
         
            // Find the maximum value in floatMat
            double maxVal;
            minMaxLoc(floatMat, nullptr, &maxVal);

            // Update the pixel intensity in zerosMat based on floatMat
            for (int y = 0; y < zerosMat.rows; ++y) {
                for (int x = 0; x < zerosMat.cols; ++x) {
                    // Calculate the change in pixel intensity based on floatMat
                    float intensityChange = floatMat.at<float>(y, x) / maxVal * 255.0;

                    // Update the pixel intensity in zerosMat
                    zerosMat.at<uchar>(y, x) = static_cast<uchar>(intensityChange);
                }
            }

            //saveMatAsCSV(zerosMat, "./resources/scaledMatrix_f.csv");
            cv::imwrite(filename, zerosMat);

        }

};


int main()
{
    //std::vector<std::tuple<long long, double, double, std::array<std::array<double, 4>, 4>, double, double, int, int>> pv_data = Project_point().load_pv_data("./resources/dump_cnr_c60/2023-12-12-105500_pv.txt");
    //Project_point().print_frame_pv(133468485008415723, "./resources/dump_cnr_c60/2023-12-12-105500_pv.txt");    
    //Project_point().print_frame_pv(133468485013080495, pv_data);    
    cout<<"start ..."<<endl;
    /*
    auto tuple_e = Project_point().extract_intrinsics("./resources/dump_cnr_c60/");

    const cv::Mat distortion = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0);
    const cv::Mat intrinsic = std::get<1>(tuple_e[0]);
    const cv::Mat p2w = std::get<2>(tuple_e[0]);
    const cv::Mat image = cv::imread("./resources/dump_cnr_c60/PV/133468485003417754.png"); //133468485003417754
    //cout << "timestamp:  "<< std::get<0>(tuple_e[0]) << endl;
    auto [ray_directions, ray_origin] = Project_point().ray_direction_per_image(intrinsic, distortion, image);
    //Project_point().print_directions_matrix(ray_directions);
    //std::cout<<ray_directions.size<<endl;
    cv::Mat ray_direction_ws = Project_point().image_to_mesh_space(p2w, ray_directions);
    cv::Mat ray_origins_ws = Project_point().image_to_mesh_space(p2w, ray_origin);
    //Project_point().print_directions_matrix(wsdir);
    
    cout<<"Building new depth"<<endl;;
    HandleMesh mesh_handler = HandleMesh("./resources/room_1st.off", 16);
   
    
    auto [hit_something_mat, hit_coords_mat, hit_distances_mat, hit_face_id_mat] = mesh_handler.project_rays(ray_origins_ws, ray_direction_ws);
    //mesh_handler.visualize_points_in_mesh(ray_origins_ws, ray_direction_ws);
    mesh_handler.saveMatAsCSV(hit_distances_mat, "./resources/hit_distance.csv");
    mesh_handler.saveMatAsCSV(hit_face_id_mat, "./resources/hit_face_id_mat.csv");
    //mesh_handler.saveMatAsCSV(hit_something_mat, "./resources/hit_something_mat.csv");
    cout<<"remapping on png"<<endl;
    // Find the minimum and maximum values in the matrix
    // Scale the values to the range [0, 255]
    
    int height = hit_distances_mat.size[0];
    int width = hit_distances_mat.size[1];

    cv::Mat scaledMatrix(height, width, CV_8UC3);
    double minVal, maxVal;
    minMaxLoc(hit_distances_mat, &minVal, &maxVal);
    cout<<"max "<<maxVal << " min "<<minVal<<endl;
    
    for(int h = 0; h < height; h++){
        for(int w = 0; w < width; w++){
            scaledMatrix.at<int>(h,w) = 255.0 / (maxVal - minVal);
        
        }
    }

    //hit_distances_mat.convertTo(scaledMatrix, CV_8UC3, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cout<<255.0 / (maxVal - minVal) <<" "<< -minVal * 255.0 / (maxVal - minVal)<<endl;
    // Save the scaled matrix as an image
    std::string outputPath = "./resources/scaled_image.png";
    cv::imwrite(outputPath, scaledMatrix);
    mesh_handler.saveFloatMatAsGrayscaleImage(hit_distances_mat, "./resources/scaled_image_fun.png");
    mesh_handler.saveMatAsCSV(scaledMatrix, "./resources/scaledMatrix.csv");

    */
    //the following i have to make into a method

    auto tuple_intrinsics = Project_point().extract_intrinsics("./resources/dump_cnr_c60/");

    const cv::Mat distortion = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0);
    HandleMesh mesh_handler = HandleMesh("./resources/room_1st.off", 16);
 
    int total_iterations = tuple_intrinsics.size();
    int progress = 0;


    omp_set_num_threads(16);
    #pragma omp parallel for 
    for (int i = 0; i < tuple_intrinsics.size(); i++){
        #pragma omp atomic
        progress++;
        const cv::Mat intrinsic = std::get<1>(tuple_intrinsics[i]);
        const cv::Mat p2w = std::get<2>(tuple_intrinsics[i]);
        string timestamp = "" + to_string(std::get<0>(tuple_intrinsics[i]));
        string path = "./resources/dump_cnr_c60/PV/"+timestamp+".png";
        const cv::Mat image = cv::imread(path);

        auto [ray_directions, ray_origin] = Project_point().ray_direction_per_image(intrinsic, distortion, image);

        cv::Mat ray_direction_ws = Project_point().image_to_mesh_space(p2w, ray_directions);
        cv::Mat ray_origins_ws = Project_point().image_to_mesh_space(p2w, ray_origin);

        auto [hit_something_mat, hit_coords_mat, hit_distances_mat, hit_face_id_mat] = mesh_handler.project_rays(ray_origins_ws, ray_direction_ws, false);
        string save_path = "./resources/new_depths/";
        mesh_handler.saveMatAsCSV(hit_distances_mat, save_path+"csv/"+timestamp+"_hit_distance.csv");
        mesh_handler.saveMatAsCSV(hit_face_id_mat, save_path+"csv/"+timestamp+"hit_face_id_mat.csv");
        mesh_handler.saveFloatMatAsGrayscaleImage(hit_distances_mat, save_path+timestamp+"_depth.png");
        
        #pragma omp critical
        std::cout<<"processed "<< progress << " out of " << total_iterations <<endl;
    }




    std::cout<<"thanks for playing cout"<<endl;

}
