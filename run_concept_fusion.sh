#!/bin/bash
#docker image build -t Paolo.Fasano/tesi_image:run_concept_fusion .
#docker run -v "$(pwd)":/tesi_image -w /tesi_image Paolo.Fasano/tesi_image:run_concept_fusion  ./run_concept_fusion.sh 2>&1 | tee debug.txt 
#docker ps -a | grep Paolo.Fasano/tesi_image:run_concept_fusion | awk '{print $1}' | xargs docker rm

: '
    This shell script calls for the concept fusion scripts 
    to transform a set of images into a pointcloud and extract the features from it.
'
only_build_image=false
run_extract_features=true
run_feature_fusion=true
run_text_demo=true
preset="cnr-60" #cf-ds = concept fusion dataset | cnr-ds = CNR dataset | "" = custom | if left balank it will use cf-ds automatically

#variables to run extract_conceptfusion_features.py
data_dir=./datasets
sequence=living_room_traj0_frei_png
checkpoint=./concept-fusion/examples/checkpoints/sam_vit_h_4b8939.pth
dataconfig_path=./concept-fusion/examples/dataconfigs/icl.yaml
device=cpu
save_dir=$data_dir/$sequence/saved-feat

#variables to run run_feature_fusion_and_save_map.py
mode=fusion
map_save_dir=$data_dir/$sequence/saved-map
image_height=120
image_width=160 
desired_featureheight=120 
desired_feature_width=160
device_ff=cuda

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
        --run_feature_fusion)
            run_feature_fusion="$2"
            shift 2
            ;;
        --run_text_demo)
            run_text_demo="$2"
            shift 2
            ;;
        --preset)
            preset="$2"
            shift 2
            ;;
        --data_dir)
            data_dir="$2"
            shift 2
            ;;
        --sequence)
            sequence="$2"
            shift 2
            ;;
        --dataconfig_path)
            dataconfig_path="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --save_dir)
            save_dir="$2"
            shift 2
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --map_save_dir)
            map_save_dir="$2"
            shift 2
            ;;
        --image_height)
            image_height="$2"
            shift 2
            ;;
        --image_width)
            image_width="$2"
            shift 2
            ;;
        --desired_feature_height)
            desired_feature_height="$2"
            shift 2
            ;;
        --desired_feature_width)
            desired_feature_width="$2"
            shift 2
            ;;
        --device_ff)
            device_ff="$2"
            shift 2
            ;;
        
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


docker image build -t Paolo.Fasano/tesi_image:run_concept_fusion .

if [[ $only_build_image == false ]]; then
    if [[ "$preset" == "cf-ds" ]]; then
        data_dir=./datasets
        sequence=living_room_traj0_frei_png
        checkpoint=./concept-fusion/examples/checkpoints/sam_vit_h_4b8939.pth
        dataconfig_path=./concept-fusion/examples/dataconfigs/icl.yaml
        device=cpu
        save_dir=$data_dir/$sequence/saved-feat

        mode=fusion
        map_save_dir=$data_dir/$sequence/saved-map
        image_height=120
        image_width=160 
        desired_feature_height=120 
        desired_feature_width=160
        device_ff=cuda

    elif [[ "$preset" == "cnr-60" ]]; then
        data_dir=./dataset
        sequence=cnr_c60
        checkpoint=./concept-fusion/examples/checkpoints/sam_vit_h_4b8939.pth
        dataconfig_path=$data_dir/$sequence/dataconfigs/icl.yaml
        device=cpu
        save_dir=$data_dir/$sequence/saved-feat

        mode=fusion
        map_save_dir=$data_dir/$sequence/saved-map
        image_height=1080
        image_width=1920
        desired_feature_height=1080
        desired_feature_width=1920
        device_ff=cpu

    elif [[ "$preset" == "cnr-ds" ]]; then
        data_dir=./datasets/cnr
        sequence=hl2-sensor-dump-i32-cnr
        checkpoint=./concept-fusion/examples/checkpoints/sam_vit_h_4b8939.pth
        dataconfig_path=./datasets/cnr/dataconfigs/icl.yaml
        device=cuda
        save_dir=$data_dir/$sequence/saved-feat

        mode=fusion
        map_save_dir=$data_dir/$sequence/saved-map 
        image_height=288
        image_width=320   
        desired_feature_height=288 
        desired_feature_width=320
        device_ff=cuda

    fi

    if [[ $run_extract_features == true ]]; then
        threshold_mb=20000
        while true; do
            echo "checking free memory"
            trap 'echo "SIGINT received, stopping loop"; exit' INT

            # Get the output of the free command
            free_output=$(free -m)
            # Extract the line containing free memory information
            available_memory_line=$(echo "$free_output" | grep "Mem")
            # Extract the free memory value from the line
            free_memory=$(echo "$available_memory_line" | awk '{print $7}')
            # Convert mb to gigabytes
            echo "Available memory is $(expr $free_memory / 1024) GB"
            echo " free_memory: $free_memory  threshold in mb: $threshold_mb"
            # Check if the available memory is lower than the threshold
            if [ "$free_memory" -gt "$threshold_mb" ]; then
                echo "Available memory is higher than $(expr $threshold_mb / 1024) GB"
                
                echo "extract_conceptfusion_features.py will run with the following parameters:"
                echo -e " --data-dir $data_dir \n --sequence $sequence \n --checkpoint-path $checkpoint \n --dataconfig-path $dataconfig_path \n --save-dir $save_dir \n --device $device \n --desired-height $desired_feature_height \n --desired-width $desired_feature_width"
                docker run -v "$(pwd)":/tesi_image -w /tesi_image Paolo.Fasano/tesi_image:run_concept_fusion python3 ./concept-fusion/examples/extract_conceptfusion_features.py --data-dir $data_dir --sequence $sequence --checkpoint-path $checkpoint --dataconfig-path $dataconfig_path --save-dir $save_dir  --device $device --desired-height $desired_feature_height --desired-width $desired_feature_width
        
                break  # Exit the loop once the condition is met
            fi
            echo "memory is less than expected, waiting for 150s"
            # Sleep for a few seconds before checking again (adjust as needed)
            sleep 150
        done
    fi

    if [[ $run_feature_fusion == true ]]; then
        threshold_2_mb=20000

        while true; do
            echo "checking free memory"
            trap 'echo "SIGINT received, stopping loop"; exit' INT

            # Get the output of the free command
            free_output=$(free -m)

            # Extract the line containing free memory information
            available_memory_line=$(echo "$free_output" | grep "Mem")

            # Extract the free memory value from the line
            free_memory=$(echo "$available_memory_line" | awk '{print $7}')

            # Convert megabytes to gigabytes
            echo "Available memory is $(expr $free_memory / 1024) GB"
            echo " free_memory: $free_memory  threshold in mb: $threshold_mb"

            # Check if the available memory is lower than the threshold
            if [ "$free_memory" -gt "$threshold_2_mb" ]; then
                echo "Available memory is higher than $(expr $threshold_2_mb / 1024) GB"
                
                echo "run_feature_fusion_and_save_map.py will run with the following parameters"
                echo -e " --mode $mode \n --dataset-path $data_dir \n --sequence $sequence \n --dataconfig-path $dataconfig_path \n --device $device_ff \n --dir-to-save-map $map_save_dir \n --image-height $image_height \n --image-width $image_width \n --desired-feature-height $desired_feature_height \n --desired-feature-width $desired_feature_width \n --feat-dir $save_dir"
                docker run -v "$(pwd)":/tesi_image -w /tesi_image Paolo.Fasano/tesi_image:run_concept_fusion python3 ./concept-fusion/examples/run_feature_fusion_and_save_map.py --mode $mode --dataset-path $data_dir --sequence $sequence --dataconfig-path $dataconfig_path --device $device_ff --dir-to-save-map $map_save_dir --image-height $image_height --image-width $image_width --desired-feature-height $desired_feature_height --desired-feature-width $desired_feature_width --feat-dir $save_dir
                
                break  # Exit the loop once the condition is met
            fi
            echo "memory is less than expected, waiting for 150s"
            # Sleep for a few seconds before checking again (adjust as needed)
            sleep 150
        done
    fi

    if [[ $run_text_demo == true ]]; then
        #load_path="./build_depth/dump_cnr_c60/pinhole_projection/saved-map-og/pointclouds"
        load_path="./build_depth/dataset/cnr_c60/saved-map/pointclouds"

        echo "demo_text_query.py will run with the following parameters:"
        echo -e " --load-path $load_path"
        docker run -v "$(pwd)":/tesi_image -w /tesi_image Paolo.Fasano/tesi_image:run_concept_fusion python3 ./concept-fusion/examples/demo_text_query.py --load-path $load_path
    fi
    #echo "running cf-manipulation.py"
    #python3 ./cf-manipulation.py
fi

docker ps -a | grep Paolo.Fasano/tesi_image:run_concept_fusion | awk '{print $1}' | xargs docker rm

cd ./build_vertex_dic
./run_vertex_dic.sh 2>&1 | tee debug.txt
