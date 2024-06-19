#include "build_vertex_dic.h"

int main(int argc, char* argv[])
{   
    Project_vertex_to_image projector = Project_vertex_to_image("./resources/dataset/room_1st.off", "./resources/dataset/");
    projector.get_vetex_to_pixel_dict("dump_cnr_c60/", "cnr_c60/depth", "cnr_c60/vertex_images_json");
}