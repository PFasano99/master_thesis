#include "./build_depth.h"

bool verbose = false;
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
void make_all_dirs(string test_mesh_path =  "./resources/ray_coord_mesh/", string save_path = "./resources/dataset/cnr_c60"){

    // Check if the directory already exists
    if (!filesystem::exists(test_mesh_path)) {
        // Create the directory
        if (filesystem::create_directory(test_mesh_path)) {
            std::cout << "Directory created successfully: " << test_mesh_path << std::endl;
        } else {
            std::cerr << "Error: Failed to create directory: "<< test_mesh_path << std::endl;
        }
    }

    // Check if the directory already exists
    if (!filesystem::exists(save_path)) {
        // Create the directory
        if (filesystem::create_directory(save_path)) {
            std::cout << "Directory created successfully: "<< save_path << std::endl;
        } else {
            std::cerr << "Error: Failed to create directory: " << save_path << std::endl;
        }
    }

    // Check if the directory already exists
    if (!filesystem::exists(save_path+"/rgb")) {
        // Create the directory
        if (filesystem::create_directory(save_path+"/rgb/")) {
            std::cout << "Directory created successfully: "<< save_path+"/rgb" << std::endl;
        } else {
            std::cerr << "Error: Failed to create directory: " << save_path+"/rgb" << std::endl;
        }
    }

    // Check if the directory already exists
    if (!filesystem::exists(save_path+"/depth/")) {
        // Create the directory
        if (filesystem::create_directory(save_path+"/depth")) {
            std::cout << "Directory created successfully: "<< save_path+"/depth" << std::endl;
        } else {
            std::cerr << "Error: Failed to create directory: " << save_path+"/depth" << std::endl;
        }
    }

    // Check if the directory already exists
    if (!filesystem::exists(save_path+"/dataconfigs")) {
        // Create the directory
        if (filesystem::create_directory(save_path+"/dataconfigs")) {
            std::cout << "Directory created successfully: "<< save_path+"/dataconfigs" << std::endl;
        } else {
            std::cerr << "Error: Failed to create directory: " << save_path+"/dataconfigs" << std::endl;
        }
    }

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

void make_gt_sim_from_extrinsics(Eigen::Matrix4d& extrinsics, const std::string& outputFile, Eigen::Matrix4d& fixed_extrinsics){
    
    std::ofstream outfile;
    outfile.open(outputFile, std::ios_base::app);
    Eigen::Matrix4d inverse_fixed_extrinsics = fixed_extrinsics.inverse();

    Eigen::Matrix4d pose = extrinsics;// * inverse_fixed_extrinsics;
    Eigen::Matrix<double, 3, 4> submatrix = pose.topRows<3>();

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


void make_dataset(int start_id = 0, int end_id = 1803, string mesh_file_path = "./resources/dataset/room_1st.off", string raw_data_path = "./resources/dataset/dump_cnr_c60/", string test_mesh_path =  "./resources/dataset/ray_coord_mesh/", string save_path = "./resources/dataset/cnr_c60" , int threads_number = 16){
    
    //create all the necessary folders
    make_all_dirs();    
    
    HandleMesh mesh_handler = HandleMesh(mesh_file_path, threads_number);
    Project_point projector = Project_point(threads_number, verbose);

    std::vector<string> cal_txt;
    std::vector<string> association;
    auto tuple_intrinsics = Project_point().extract_intrinsics(raw_data_path);

    float done_images = start_id;

    Eigen::Matrix4d fixed_extrinsics = Project_point().read_fixed_extrinsics(raw_data_path+"Depth Long Throw_extrinsics.txt");

    omp_set_num_threads(threads_number);
    #pragma omp parallel for ordered
    for (int i = start_id; i < end_id; i+=1){
        //tuple_intrinsics.size()
        Eigen::Matrix3d intrinsic = std::get<1>(tuple_intrinsics[i]);
        Eigen::Matrix4d extrinsic = std::get<2>(tuple_intrinsics[i]);
        
        string timestamp = "" + to_string(std::get<0>(tuple_intrinsics[i]));

        #pragma omp ordered
        make_gt_sim_from_extrinsics(extrinsic, save_path+"/odometry.gt.sim", fixed_extrinsics);

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

        #pragma omp ordered
        association.push_back(std::to_string(i)+" "+ save_path+"/depth/"+timestamp+"_depth.png " + std::to_string(i)+" "+save_path+"/rgb/"+timestamp+".png ");
        #pragma omp orederd
        cal_txt.push_back(timestamp + " " +  std::to_string(intrinsic(0,0)) + " " + std::to_string(intrinsic(1,1)) +" "+ std::to_string(intrinsic(0,2)) +" "+std::to_string(intrinsic(1,2)));

        auto [ray_origin, ray_directions] = projector.ray_direction_per_image(intrinsic);
                
        std::vector<std::vector<vcg::Point3f>> ray_origin_ws = projector.image_to_mesh_space(extrinsic, ray_origin);    
        std::vector<std::vector<vcg::Point3f>> ray_direction_ws = projector.image_to_mesh_space(extrinsic, ray_directions);    

        vcg::Point3f origin = ray_origin_ws[0][0];

        int rows = ray_direction_ws.size();
        int cols = ray_direction_ws[0].size();

        std::vector<std::vector<vcg::Point3f>> direction(2, std::vector<vcg::Point3f>(2));
        direction[0][0] = ray_direction_ws[0][0];
        direction[0][1] = ray_direction_ws[0][cols-1];
        direction[1][0] = ray_direction_ws[rows-1][0];
        direction[1][1] = ray_direction_ws[rows-1][cols-1];

        mesh_handler.visualize_points_in_mesh(origin, direction,test_mesh_path+timestamp+".ply", true);
        
        auto [hit_something_mat, hit_coords_mat, hit_distances_mat, hit_face_id_mat] = mesh_handler.project_rays(ray_origin, ray_direction_ws, false, false);

        //mesh_handler.saveMatAsCSV(hit_distances_mat, save_path+"depth/"+"csv/"+timestamp+"_hit_distance.csv");
        //mesh_handler.saveMatAsCSV(hit_face_id_mat, save_path+"depth/"+"csv/"+timestamp+"hit_face_id_mat.csv");
        mesh_handler.saveFloatMatAsGrayscaleImage(hit_distances_mat, save_path+"/depth/"+timestamp+"_depth.png");
        
        #pragma opm critical
        done_images+=1;

        #pragma omp critical
        std::cout<<"processed "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;
    }
    cout<<""<<endl;
    save_to_txt(cal_txt, save_path, "/calibration.txt", true);
    save_to_txt(association, save_path, "/association.txt", true);
    //make_gt_sim(raw_data_path+"pinhole_projection/odometry.log", save_path+"/odometry.gt.sim");
    make_yaml_file(save_path, std::get<1>(tuple_intrinsics[1]),"/dataconfigs/icl.yaml");
}


int main(int argc, char* argv[])
{   
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

}