: '
    The run_build_depth.sh has two parameters if not set it will both build and run:
        - build_only, if true the script will only build the docker image; 
        - run_only, if true the script will only run from a previously build image;
        - delate_container, if true all the container from the image will be delated;
'
build_only=true
run_only=true
delate_container=true

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
    docker image build -t Paolo.Fasano/tesi_image:cpp .  #--no-cache
    delate_container=false
fi

if [[ "$run_only" == true ]]; then
    cd ../
    path_to_data="$(pwd)"
    cd ./build_depth

    c=0

    # Loop over the range of numbers from 1 to 1800
    for ((i = 0; i <= 1800; i+=900)); do
        trap 'echo "SIGINT received, stopping loop"; exit' INT
        # Call the compiled C++ program with the current number as argument
        docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp ./build_depth/build/build_depth "$c" "$i"
        c=$i
    done

    docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp ./build_depth/build/build_depth 1800 1803


    #docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp ./build_depth/build/build_depth
   
    if [[ "$delate_container" == true ]]; then
        echo "Delating container of image Paolo.Fasano/tesi_image:cpp"
        docker ps -a | grep Paolo.Fasano/tesi_image:cpp | awk '{print $1}' | xargs docker rm
    fi

    
fi