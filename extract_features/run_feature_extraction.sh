only_build_image=false
run_extract_features=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only_build_image)
            only_build_image="$2"
            shift 2
            ;;
        --run_extract_features)
            run_extract_features="$2"
            shift 2
            ;;        
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


docker image build -t Paolo.Fasano/tesi_image:extract_features .


if [[ $only_build_image == false ]]; then

    if [[ $run_extract_features == true ]]; then
        parentdir="$(dirname "$(pwd)")"
        echo $parentdir
        #docker run -v "$parentdir":/workspace/resources Paolo.Fasano/tesi_image:extract_features ls resources
        docker run -it -v "$parentdir":/workspace/resources Paolo.Fasano/tesi_image:extract_features python3 ./resources/extract_features/feature_extraction.py
    fi

fi