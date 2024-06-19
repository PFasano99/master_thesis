docker image build -t Paolo.Fasano/tesi_image:cpp .  #--no-cache

cd ../
path_to_data="$(pwd)"
cd ./build_depth

c=0

# Loop over the range of numbers from 1 to 1800
for ((i = 0; i <= 1800; i+=50)); do
    trap 'echo "SIGINT received, stopping loop"; exit' INT
    # Call the compiled C++ program with the current number as argument
    docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp ./build_depth/build/build_depth "$c" "$i"
    c=$i
done

docker run -v "$(pwd)":/workspace/builded_cpp -v $path_to_data:/workspace/resources Paolo.Fasano/tesi_image:cpp ./build_depth/build/build_depth 1800 1803

docker ps -a | grep Paolo.Fasano/tesi_image:cpp | awk '{print $1}' | xargs docker rm


#/home/paolo.fasano/tesi_fasano/build_depth