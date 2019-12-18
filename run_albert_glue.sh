#!/bin/bash
set -e
set -x

export TASK=CoLA
export VERSION=1
export ALBERT_DIR=base

export CURRENT_PWD=/home/ubuntu
export GLUE_DIR=${CURRENT_PWD}/glue_data
export OUTPUT_DIR=${CURRENT_PWD}/sp_output/${TASK}_${ALBERT_DIR}_v${VERSION}
export LOG_DIR=own
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir $OUTPUT_DIR
fi

export BS=8
export MSL=512
export LR=1e-05
export WPSP=630
export TSP=10672

pip3 install numpy
pip3 install -r requirements.txt

set +x

#sudo CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
sudo python3 -m albert.run_multigpus_sp \
    --do_train=True \
    --do_eval=True \
    --data_dir=${GLUE_DIR} \
    --cached_dir=${CURRENT_PWD}/cached_albert_tfrecord \
    --task_name=${TASK} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${MSL} \
    --train_step=${TSP} \
    --warmup_step=${WPSP} \
    --train_batch_size=${BS} \
    --learning_rate=${LR} \
    --num_gpu_cores=8 \
    --strategy_type=mirror \
    --pretrained_model_dir=${CURRENT_PWD}/albert_model/${ALBERT_DIR}_v${VERSION}/ \
    2>&1 | tee ${OUTPUT_DIR}/${LOG_DIR}.log
