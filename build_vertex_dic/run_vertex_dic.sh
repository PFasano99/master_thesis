: '
    The run_vertex_dic.sh has two parameters if not set it will both build and run:
        - build_depth_image, if true the script will build the depth image (to use if you have made some changes to build_depth.h)
        - build_only, if true the script will only build the docker image; 
        - run_only, if true the script will only run from a previously build image;

'
build_depth_image=false
build_only=false
build_image=true
run_cpp=true
run_py=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build_only)
            build_only="$2"
            shift 2
            ;;
        --build_image)
            build_only="$2"
            shift 2
            ;;
        --run_py)
            run_py="$2"
            shift 2
            ;;
        --run_cpp)
            run_cpp="$2"
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

if [[ "$build_only" == true ]]; then
    echo "Building only image Paolo.Fasano/tesi_image:cpp_vertex_dic"
    run_cpp=false
    run_py=false
    build_image=true
fi

if [[ "$build_depth_image" == true ]]; then
    echo "Building Paolo.Fasano/tesi_image:cpp"
    cd ../build_depth
    ./run_build_depth.sh --build_only true
    cd ../build_vertex_dic
else
    echo "Using prexisting Paolo.Fasano/tesi_image:cpp"
fi

if [[ "$build_image" == true ]]; then
    echo "Building new image Paolo.Fasano/tesi_image:cpp_vertex_dic"
    docker image build -t Paolo.Fasano/tesi_image:cpp_vertex_dic .  #--no-cache
else
    echo "Using prexisting image Paolo.Fasano/tesi_image:cpp_vertex_dic"
fi

if [[ "$build_only" == false ]]; then
    cd ../
    path_to_data="$(pwd)"
    cd ./build_depth
    if [[ "$run_cpp" == true ]]; then
        docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp_vertex_dic ./build_vertex_dic/build/build_vertex_dic
    else
        echo "skipping cpp"
    fi
    
    if [[ "$run_py" == true ]]; then
        docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp_vertex_dic python3 ./build_vertex_dic/bin_to_png.py
    else
        echo "skipping py"
    fi

fi


docker ps -a | grep Paolo.Fasano/tesi_image:cpp_vertex_dic | awk '{print $1}' | xargs docker rm
