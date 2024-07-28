: '
    The run_vertex_dic.sh has two parameters if not set it will both build and run:
        - build_depth_image, if true the script will build the depth image (to use if you have made some changes to build_depth.h)
        - build_vertex_image, if true the script will build the vertex image (to use if you have made some changes to build_vertex_dic.h)
        - build_only, if true the script will only build the docker image; 
        - build_image, if true the script will build the dockerFile 
        - run_cpp, if true the script will run the cpp;
        - run_py, if true the script will run the py;
        - delate_container, if true all the container from the image will be delated;
'
build_depth_image=false
build_vertex_image=false
build_only=false
build_image=true
run_cpp=true
run_py=true
delate_container=true

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
        --run_cpp)
            run_cpp="$2"
            shift 2
            ;;
        --run_py)
            run_py="$2"
            shift 2
            ;;
        --build_depth_image)
            build_depth_image="$2"
            shift 2
            ;;
        --build_vertex_image)
            build_vertex_image="$2"
            shift 2
            ;;
        --delate_container)
            delate_container="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ "$build_only" == true ]]; then
    echo "Building only image Paolo.Fasano/tesi_image:cpp_demo_query"
    run_cpp=false
    build_image=true
    delate_container=false
fi

if [[ "$build_depth_image" == true ]]; then
    echo "Building Paolo.Fasano/tesi_image:cpp"
    cd ../build_depth
    ./run_build_depth.sh --build_only true
    cd ../demo_query
else
    echo "Using prexisting Paolo.Fasano/tesi_image:cpp"
fi

if [[ "$build_vertex_image" == true ]]; then
    echo "Building Paolo.Fasano/tesi_image:cpp_vertex_dic"
    cd ../build_vertex_dic
    ./run_vertex_dic.sh --build_only true
    cd ../build_vertex_dic
else
    echo "Using prexisting Paolo.Fasano/tesi_image:cpp_vertex_dic"
fi

if [[ "$build_image" == true ]]; then
    echo "Building new image Paolo.Fasano/tesi_image:cpp_demo_query"
    docker image build -t Paolo.Fasano/tesi_image:cpp_demo_query .  #--no-cache
else
    echo "Using prexisting image Paolo.Fasano/tesi_image:cpp_demo_query"
fi

if [[ "$build_only" == false ]]; then
    cd ../
    path_to_data="$(pwd)"
    cd ./build_depth

    if [[ "$run_py" == true ]]; then
        docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp_demo_query python3 ./demo_query/text_to_features.py
    else
        echo "skipping py"
    fi

    if [[ "$run_cpp" == true ]]; then
        docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp_demo_query ./demo_query/build/test_feat
    else
        echo "skipping cpp"
    fi
fi

if [[ "$delate_container" == true ]]; then
    echo "Delating container of image Paolo.Fasano/tesi_image:cpp_demo_query"
    docker ps -a | grep Paolo.Fasano/tesi_image:cpp_demo_query | awk '{print $1}' | xargs docker rm
fi


