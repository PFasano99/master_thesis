#include "./build_depth.h"

bool verbose = false;

void create_directory_if_not_exists(const std::string& path) {
    if (!filesystem::exists(path)) {
        if (filesystem::create_directories(path)) {
            std::cout << "Directory created successfully: " << path << std::endl;
        } else {
            std::cerr << "Error: Failed to create directory: " << path << std::endl;
        }
    }
}

/*
    @param string test_mesh_path: the path to save the generated control meshes 
    @param string save_path: the path to save the generated folders

    @description This method creates the folder structure of the dataset as desired by the Concept fusion model (https://github.com/concept-fusion/concept-fusion)
        The folder structure will be the following 
        dataset
            |_ dataset_name
                |_ dataconfigs
                |_ depth
                |_ rgb
*/
void make_all_dirs(const string test_mesh_path =  "./resources/dataset/ray_coord_mesh/", const string save_path = "./resources/dataset/cnr_c60"){

    create_directory_if_not_exists(test_mesh_path);
    create_directory_if_not_exists(save_path + "/rgb");
    create_directory_if_not_exists(save_path + "/depth");
    create_directory_if_not_exists(save_path + "/dataconfigs");

}

/*
    @param std::vector<string> elements: a vector of all the elements in string format to be saved
    @param string file_path: the path to save the .txt file
    @param string file_name: the name of the file to be saved
    @param bool append: if true and the file already exists the new values are saved appended at the bottom of the file
*/
void save_to_txt(std::vector<string> elements, string file_path, string file_name, bool append = false){
    string filename = file_path+file_name;

    ofstream file;

    if (append)
        file.open(filename, std::ios_base::app);
    else
        file.open(filename, std::ios_base::out);

    if (!file.is_open()) {
        cerr << "Error: Unable to open file for writing! "<< file_path+file_name << endl;
    }
    else{
        for (int i = 0; i < elements.size(); i++) {
            file << elements[i];
            file << endl;
        }
        // Close the file
        file.close();
        if (verbose)
            std::cout << "Data saved to file successfully: "<< file_path+file_name << std::endl;
    }

}

// Function to process the file odomotry.log and save the values to odometry.gt.sim
/*
    @param std::string& inputFile: the path to the input file (odometry.log)
    @param std::string& outputFile: the path to save the generated odometry.gt.sim

    
*/
void make_gt_sim_from_odometry(const std::string& inputFile, const std::string& outputFile) {
    std::ifstream fin(inputFile); // Open the input file in read mode
    if (!fin.is_open()) {
        std::cerr << "Error: Unable to open input file." << std::endl;
        return;
    }

    std::ofstream fout(outputFile); // Open the output file in write mode
    if (!fout.is_open()) {
        std::cerr << "Error: Unable to open output file." << std::endl;
    }
    
    std::vector<string> all_lines;
    std::string line;
    int count = 0;

    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }
        all_lines.push_back(line);
    }

    for(int i = 1; i< all_lines.size(); i+=5){
        fout << all_lines[i]+"\n";
        fout << all_lines[i+1]+"\n";
        fout << all_lines[i+2]+"\n \n";
    }

    fin.close();
    fout.close();
    if (verbose)
        std::cout << "Data rearranged and saved to file successfully: "<< outputFile << std::endl;
}

void make_gt_sim_from_extrinsics(const Eigen::Matrix4d& extrinsics, const std::string& outputFile){
    
    std::ofstream outfile;
    outfile.open(outputFile, std::ios_base::app);

    Eigen::Matrix<double, 3, 4> submatrix = extrinsics.topRows<3>();

    if (outfile.is_open()) {
        outfile << submatrix << std::endl;
        outfile << std::endl;
        // Close the file
        outfile.close();
        
        if (verbose)
            std::cout << "Data rearranged and saved to file successfully: "<< outputFile << std::endl;
    } 
    else {
        std::cerr << "Unable to open file for writing" << std::endl;
    }
    
}

void make_yaml_file(string file_path, Eigen::Matrix3d intrinsic, string file_name, string dataset_name = "icl", int image_height= 1080, int image_width=1920, float png_depth_scale = 5000, int crop_edge = 0){
    
    string filename = file_path+file_name;
    ofstream file(filename, std::ios_base::out);

    if (!file.is_open()) {
        cerr << "Error: Unable to open file for writing! "<< file_path+file_name << endl;
    }
    else{
        string element = "dataset_name: '"+dataset_name+"' \ncamera_params: \n  image_height: "+std::to_string(image_height)+"\n  image_width: "+std::to_string(image_width)+"\n  fx: "+std::to_string(intrinsic(0,0))+"\n  fy: "+std::to_string(intrinsic(1,1))+"\n  cx: "+std::to_string(intrinsic(0,2))+"\n  cy: "+std::to_string(intrinsic(1,2))+"\n  png_depth_scale: "+std::to_string(png_depth_scale)+"\n  crop_edge: "+std::to_string(crop_edge);
        file << element;
        file.close();
        if (verbose)
            std::cout << "Data saved to file successfully: "<< file_path+file_name << std::endl;
    }

}


void make_dataset(int start_id, int end_id, string mesh_file_path = "./resources/dataset/room_1st.off", string raw_data_path = "./resources/dataset/dump_cnr_c60/", string test_mesh_path =  "./resources/dataset/ray_coord_mesh/", string save_path = "./resources/dataset/cnr_c60" , int threads_number = 64){
    
    //create all the necessary folders
    make_all_dirs();    
    
    HandleMesh mesh_handler = HandleMesh(mesh_file_path, threads_number);
    Project_point projector = Project_point(threads_number, verbose);

    auto tuple_intrinsics = Project_point().extract_intrinsics(raw_data_path);
    std::vector<string> cal_txt(tuple_intrinsics.size(),"");
    std::vector<string> association(tuple_intrinsics.size(),"");
    
    float done_images = start_id;
    std::cout << "processed " << std::setprecision(3) << std::fixed << done_images / tuple_intrinsics.size() * 100 << "% | " << static_cast<int>(done_images) << "/" << tuple_intrinsics.size() << "\r" << std::flush;
    omp_set_num_threads(threads_number);
    #pragma omp parallel for 
    for (int i = done_images; i < end_id; i+=1){
        //tuple_intrinsics.size()
        string timestamp = "" + to_string(std::get<0>(tuple_intrinsics[i]));
        Eigen::Matrix3d intrinsic = std::get<1>(tuple_intrinsics[i]);
        Eigen::Matrix4d extrinsic = std::get<2>(tuple_intrinsics[i]);
        
        //copying all the rgb.png files to the new dataset
        try {
            // Check if the source file exists
            if (!filesystem::exists(raw_data_path+"PV/"+timestamp+".png")) {
                std::cerr << "Error: Source file does not exist: " << raw_data_path+"PV/"+timestamp+".png" << std::endl;
            }
            // Copy the file
            filesystem::copy_file(raw_data_path+"PV/"+timestamp+".png", save_path+"/rgb/"+timestamp+".png", filesystem::copy_options::overwrite_existing);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        association[i] = (std::to_string(i)+" "+ save_path+"/depth/"+timestamp+"_depth.png " + std::to_string(i)+" "+save_path+"/rgb/"+timestamp+".png ");
        cal_txt[i] = (timestamp + " " +  std::to_string(intrinsic(0,0)) + " " + std::to_string(intrinsic(1,1)) +" "+ std::to_string(intrinsic(0,2)) +" "+std::to_string(intrinsic(1,2)));
        
        auto [ray_origin, ray_directions] = projector.ray_direction_per_image(intrinsic);
        
        std::vector<std::vector<vcg::Point3f>> ray_origin_ws(ray_origin.size(), std::vector<vcg::Point3f>(ray_origin[0].size())); 
        projector.image_to_mesh_space(ray_origin_ws, extrinsic, ray_origin);    

        std::vector<std::vector<vcg::Point3f>> ray_direction_ws(ray_directions.size(), std::vector<vcg::Point3f>(ray_directions[0].size()));
        projector.image_to_mesh_space(ray_direction_ws, extrinsic, ray_directions);    

        //create_directory_if_not_exists(save_path+"/depth/"+"csv");
        //mesh_handler.saveMatAsCSV(ray_origin_ws, save_path+"/depth/"+"csv/"+timestamp+"_ray_origin.csv");
        //mesh_handler.saveMatAsCSV(ray_direction_ws, save_path+"/depth/"+"csv/"+timestamp+"_ray_direction.csv");
        
        vcg::Point3f origin = ray_origin_ws[0][0];

        int rows = ray_direction_ws.size();
        int cols = ray_direction_ws[0].size();

        auto [hit_something_mat, hit_coords_mat, hit_distances_mat, hit_face_id_mat] = mesh_handler.project_rays(ray_origin_ws, ray_direction_ws, false, false);
        
        /*
        std::vector<std::vector<vcg::Point3f>> hp_direction(2, std::vector<vcg::Point3f>(2));
        hp_direction[0][0] = hit_coords_mat[0][0];
        hp_direction[0][1] = hit_coords_mat[0][cols-1];
        hp_direction[1][0] = hit_coords_mat[rows-1][0];
        hp_direction[1][1] = hit_coords_mat[rows-1][cols-1];

        //mesh_handler.visualize_points_in_mesh(origin, hp_direction, test_mesh_path+timestamp+"hp_d.ply", true, 10);
        //mesh_handler.visualize_points_in_mesh(origin, hit_coords_mat, test_mesh_path+timestamp+"_hp_nd.ply", true, 1, true);
        */
        //mesh_handler.saveMatAsCSV(hit_distances_mat, save_path+"/depth/"+"csv/"+timestamp+"_hit_distance.csv");
        //mesh_handler.saveMatAsCSV(hit_face_id_mat, save_path+"/depth/"+"csv/"+timestamp+"hit_face_id_mat.csv");
        
        mesh_handler.saveFloatMatAsGrayscaleImage(hit_distances_mat, save_path+"/depth/"+timestamp+"_depth.png", 5000);
        mesh_handler.saveFloatMatAsGrayscaleImage(hit_distances_mat, save_path+"/depth/"+timestamp+"_depth.pfm");
        
        #pragma omp atomic
        done_images += 1;

        //if (int(done_images) % 5 == 0) {  // Print every 10 images
            #pragma omp critical
            {
                std::cout << "processed " << std::setprecision(3) << std::fixed << done_images / tuple_intrinsics.size() * 100 << "% | " << static_cast<int>(done_images) << "/" << tuple_intrinsics.size() << "\r" << std::flush;
            }
        //}
    }

    for (int i = 0; i < tuple_intrinsics.size(); i+=1){
        Eigen::Matrix4d extrinsic = std::get<2>(tuple_intrinsics[i]);
        make_gt_sim_from_extrinsics(extrinsic, save_path+"/odometry.gt.sim");
    }

    cout<<""<<endl;
    save_to_txt(cal_txt, save_path, "/calibration.txt", true);
    save_to_txt(association, save_path, "/association.txt", true);
    make_yaml_file(save_path, std::get<1>(tuple_intrinsics[1]),"/dataconfigs/icl.yaml");
}

int main(int argc, char* argv[])
{   
    /*
    cout<<"size of argc "<< argc <<endl;
    if(argc != 0)
        verbose = argv[0];

    for(int i = 0; i< argc; i++)
        cout<<"argv[i] "<< argv[i] << endl;
    */
    cout << "Making dataset..." << endl;
    auto start = high_resolution_clock::now();

    //make_dataset();
    
    int start_id = std::atoi(argv[1]);
    int end_id = std::atoi(argv[2]);
    cout<<"size of argc "<< argc <<endl;
    if(argc > 3)
        verbose = argv[3];

    for(int i = 0; i< argc; i++)
        cout<<"argv[i] "<< argv[i] << endl;

    cout<<"start range: "<< start_id << " to "<< end_id <<endl;
    make_dataset(start_id, end_id);
    cout<<"done range: "<< start_id << " to "<< end_id <<endl;

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    cout << endl << "Making dataset took " << elapsed.count() << " seconds" << endl;
}