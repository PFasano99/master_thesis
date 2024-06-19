docker image build -t Paolo.Fasano/tesi_image:cpp_vertex_dic .  #--no-cache

cd ../
path_to_data="$(pwd)"
cd ./build_depth

c=0

# Loop over the range of numbers from 1 to 1800
#for ((i = 0; i <= 1800; i+=50)); do
#    trap 'echo "SIGINT received, stopping loop"; exit' INT
#    # Call the compiled C++ program with the current number as argument
#    docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp_vertex_dic ./build_depth/build/build_depth "$c" "$i"
#    c=$i
#done

docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp_vertex_dic ./build_vertex_dic/build/build_vertex_dic 1800 1803

docker ps -a | grep Paolo.Fasano/tesi_image:cpp_vertex_dic | awk '{print $1}' | xargs docker rm
#docker run -v "$(pwd)":/workspace/ Paolo.Fasano/tesi_image:cpp_vertex_dic ls

#/home/paolo.fasano/tesi_fasano/build_depth