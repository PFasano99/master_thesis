# Paolo Fasano - Master thesis


## Installation guide
The following section will describe the steps needed to install this repository and run the content.
### Manual installation

If you want to manually install all the needed dependencies in orders run this repository you will need to install:
<b>Python requirements:</b> 

        git clone https://github.com/PFasano99/master_thesis
        pip install -r requirements.txt

**concept-fusion** Install `concept-fusion` following the [instructions here](https://github.com/concept-fusion/concept-fusion/tree/main)

**gradslam:** in the concept fusion branch

        git clone https://github.com/gradslam/gradslam.git
        cd gradslam
        git checkout conceptfusion
        pip install -e .
    
**segment-anything**: Install `segment-anything` by following [instructions here](https://github.com/facebookresearch/segment-anything).

**openclip**: Install `openclip` following [instructions here](https://github.com/mlfoundations/open_clip).


### Dockerfile installation
If you want to use the Dockerfile provided to install all the needed dependencies in orders run this repository you will need:

**Docker** install Docker following the [instructions here](https://docs.docker.com/get-docker/)

<b>Run Dockerfile:</b> 

        git clone https://github.com/PFasano99/master_thesis
        docker image build -t your_username/tesi_image:v0.1 . - < Dockerfile

For any errors in the installation of docker and/or the creation/use of a docker image refer to the [official docker docs](https://docs.docker.com/reference/). 

## Run concept-fusion
In order to run concept-fusion on a dataset the official instructions can be found in the [concept-fusion git](https://github.com/concept-fusion/concept-fusion/tree/main)

I provide run_concept_fusion.sh to run the scripts in one single solution.

        docker --runtime nvidia run --rm -it -v /home/my_username/my_project:/tesi_image -w /tesi_image your_username/tesi_image:v0.1  ./run_concept_fusion.sh [options]

        A list of all the options can be found in the .sh file

---

**This readme is incomplete and will expanded as the projects moves forward**