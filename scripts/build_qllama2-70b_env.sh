#!/bin/bash

# define env. variables
model_name=qllama2-70b
model_dir=language/llama2-70b
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)
quant_data_dir=$data_dir/quantization/llama2-70b
tag=MLPerf4.1-v3.13
quant_data_dvc_dir=quantized/LLaMA2-70B/mlperf_submission/W8A8KV8/80L

# work on model directory
cd $work_dir

# create and enter conda env.
printf "\n============= STEP-1: Create conda environment and activate =============\n"
conda remove -n $env_name --all -y
rm -rf $conda_base/env/$env_name
conda env create -f $git_dir/scripts/envs/$model_name\_env.yml
set +u
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name
set -u

# build mlperf loadgen
printf "\n============= STEP-2: Build mlperf loadgen =============\n"
pip install pybind11==2.10.4
cd $git_dir/loadgen; python setup.py install
cd -

# pull model and dataset
printf "\n============= STEP-3: Pull dvc data =============\n"
pip install dvc[s3]
dvc pull $data_dir/models/llama2/Llama-2-70b-chat-hf/model-0000*.dvc --force
dvc pull $data_dir/models/llama2/Llama-2-70b-chat-hf/model-0001*.dvc --force
# dvc pull $data_dir/models/llama2/Llama-2-70b-chat-hf/pytorch_model-0000*.dvc --force
# dvc pull $data_dir/models/llama2/Llama-2-70b-chat-hf/pytorch_model-0001*.dvc --force
shopt -s extglob
dvc pull $data_dir/models/llama2/Llama-2-70b-chat-hf/!(model-000*|pytorch_model-000*).dvc --force
dvc pull $data_dir/dataset/open-orca/validation --force
dvc pull $data_dir/dataset/open-orca/calibration --force
dvc pull $data_dir/quantization/llama2-70b.dvc --force

# pull quantization 
printf "\n============= STEP-4: Pull quantization data =============\n"
cd $git_dir
git clone https://github.com/furiosa-ai/furiosa-llm-models-artifacts.git
cd $git_dir/furiosa-llm-models-artifacts
git checkout $tag

dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml.dvc -r origin --force
dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy.dvc -r origin --force

mkdir -p $quant_data_dir/calibration_range
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml $quant_data_dir/calibration_range/quant_format.yaml
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy $quant_data_dir/calibration_range/quant_param.npy



printf "\n============= End of build =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
