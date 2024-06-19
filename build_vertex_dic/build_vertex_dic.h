#include "../build_depth/build_depth.h"

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
        void get_vetex_to_pixel_dict(string path_to_pv){
            
            Project_point projector = Project_point(n_threads);
            HandleMesh mesh_handler = HandleMesh(path_to_mesh, n_threads, verbose);
            
            //
            path_to_pv = path_to_dataset+path_to_pv;
            auto tuple_intrinsics = projector.extract_intrinsics(path_to_pv);
            float done_images = 1;
            omp_set_num_threads(n_threads);
            #pragma omp parallel for ordered
            for (int i = 1; i < 10; i+=1){
                //tuple_intrinsics.size()
                Eigen::Matrix3d intrinsic = std::get<1>(tuple_intrinsics[i]);
                Eigen::Matrix4d extrinsic = std::get<2>(tuple_intrinsics[i]);
                
                string timestamp = "" + to_string(std::get<0>(tuple_intrinsics[i]));
                cout<<timestamp<<endl;
                #pragma omp critical
                std::cout<<"processed "<<std::setprecision(3) << std::fixed<< done_images/tuple_intrinsics.size()*100 << "% | "<<static_cast<int>(done_images)<<"/"<<tuple_intrinsics.size()<<"\r" << std::flush;
                done_images++;
            }
            cout<<""<<endl;

        }

};
