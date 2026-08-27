#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

python ./script/update_embodiment_config_path.py > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config
rm -rf data/${task_name}/${task_config}/.cache
