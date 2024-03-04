podman image build -t Paolo.Fasano/tesi_image:cpp .  #--no-cache
podman run -v "$(pwd)":/workspace/resources Paolo.Fasano/tesi_image:cpp ./build_depth-cpp
