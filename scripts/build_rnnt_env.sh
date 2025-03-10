#!/bin/bash

# define env. variables
model_name=rnnt
model_dir=speech_recognition/rnnt
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)

# work on model directory
cd $work_dir

# create and enter conda env.
printf "\n============= STEP-1: Create conda environment and activate =============\n"
conda remove -n $env_name --all -y
rm -rf $conda_base/envs/$env_name
conda env create -f $git_dir/scripts/envs/$model_name\_env.yml
# conda env create -f environment.yml
set +u
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name
set -u

# build mlperf loadgen
printf "\n============= STEP-2: Build mlperf loadgen =============\n"
pip install pybind11==2.11.1
cd $git_dir/loadgen; python setup.py install
cd -

# pull model and dataset
printf "\n============= STEP-3: Pull dvc data =============\n"
pip install dvc[s3]
dvc pull $data_dir/models/rnnt --force
dvc pull $data_dir/dataset/librispeech --force

printf "\n============= End of build =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir


