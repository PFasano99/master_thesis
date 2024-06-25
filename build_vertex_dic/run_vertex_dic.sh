: '
    The run_vertex_dic.sh has two parameters if not set it will both build and run:
        - build_depth_image, if true the script will build the depth image (to use if you have made some changes to build_depth.h)
        - build_only, if true the script will only build the docker image; 
        - run_only, if true the script will only run from a previously build image;

'
build_depth_image=false
build_only=false
run_only=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build_only)
            build_only="$2"
            shift 2
            ;;
        --run_only)
            run_only="$2"
            shift 2
            ;;
        --build_depth_image)
            build_depth_image="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ "$build_depth_image" == true ]]; then
    cd ../build_depth
    ./run_build_depth.sh --build_only true
    cd ../build_vertex_dic
fi

if [[ "$run_only" == false ]]; then
    docker image build -t Paolo.Fasano/tesi_image:cpp_vertex_dic .  #--no-cache
fi

if [[ "$build_only" == false ]]; then
    cd ../
    path_to_data="$(pwd)"
    cd ./build_depth

    docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp_vertex_dic ./build_vertex_dic/build/build_vertex_dic

    docker ps -a | grep Paolo.Fasano/tesi_image:cpp_vertex_dic | awk '{print $1}' | xargs docker rm
fi