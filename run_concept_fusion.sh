#!/bin/bash

#
: '
    This shell script calls for the concept fusion scripts 
    to transform a set of images into a pointcloud and extract the features from it.
'

preset="" #cf-ds = concept fusion dataset | cnr-ds = CNR dataset | "" = custom | if left balank it will use cf-ds automatically

#variables to run extract_conceptfusion_features.py
data_dir=./datasets
sequence=living_room_traj0_frei_png
checkpoint=./concept-fusion/examples/checkpoints/sam_vit_h_4b8939.pth
dataconfig_path=./concept-fusion/examples/dataconfigs/icl.yaml
device=cpu
save_dir=$data_dir/$sequence/saved-feat

#variables to run run_feature_fusion_and_save_map.py
mmode=fusion
map_save_dir=$data_dir/$sequence/saved-map
image_height=120
image_width=160 
desired_featureheight=120 
desired_feature_width=160
device_ff=cuda

while [[ $# -gt 0 ]]; do
    case "$1" in
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
    desired_featureheight=120 
    desired_feature_width=160
    device_ff=cuda

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
    desired_featureheight=288 
    desired_feature_width=320
    device_ff=cuda
elif [[ "$preset" == "cnr-ds-or" ]]; then
    data_dir=./datasets/cnr
    sequence=hl2-sensor-dump-i32-cnr_original_images
    checkpoint=./concept-fusion/examples/checkpoints/sam_vit_h_4b8939.pth
    dataconfig_path=$data_dir/$sequence/dataconfigs/icl.yaml
    device=cpu
    save_dir=$data_dir/$sequence/saved-feat

    mode=fusion
    map_save_dir=$data_dir/$sequence/saved-map 
    image_height=760
    image_width=428   
    desired_featureheight=760 
    desired_feature_width=428
    device_ff=cuda
fi

echo "extract_conceptfusion_features.py will run with the following parameters:"
echo -e " --data-dir $data_dir \n --sequence $sequence \n --checkpoint-path $checkpoint \n --dataconfig-path $dataconfig_path \n --save-dir $save_dir \n --device $device"
python3 ./concept-fusion/examples/extract_conceptfusion_features.py --data-dir $data_dir --sequence $sequence --checkpoint-path $checkpoint --dataconfig-path $dataconfig_path --save-dir $save_dir  --device $device

echo "run_feature_fusion_and_save_map.py will run with the following parameters"
echo -e " --mode $mode \n --dataset-path $data_dir \n --sequence $sequence \n --dataconfig-path $dataconfig_path \n --device $device_ff \n --dir-to-save-map $map_save_dir \n --image-height $image_height \n --image-width $image_width \n --desired-feature-height $desired_feature_height \n --desired-feature-width $desired_feature_width \n --feat-dir $save_dir"
python3 ./concept-fusion/examples/run_feature_fusion_and_save_map.py --mode $mode --dataset-path $data_dir --sequence $sequence --dataconfig-path $dataconfig_path --device $device_ff --dir-to-save-map $map_save_dir --image-height $image_height --image-width $image_width --desired-feature-height $desired_feature_height --desired-feature-width $desired_feature_width --feat-dir $save_dir