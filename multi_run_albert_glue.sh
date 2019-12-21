#!/bin/bash
set -e
set -x

export TASK=CoLA
export ALBERT_DIR=base
export VERSION=2
export CURRENT_PWD=/home/ubuntu

export GLUE_DIR=${CURRENT_PWD}/glue_data
export OUTPUT_DIR=${CURRENT_PWD}/albert_output/${TASK}_${ALBERT_DIR}_v${VERSION}

export BS=8
export MSL=128
export LR=5e-06
export WPSP=320
export TSP=5336

pip3 install numpy
pip3 install -r requirements.txt

set +x

sudo CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 -m albert.run_multigpus_classifier \
    --do_train=True \
    --do_eval=True \
    --strategy_type=mirror \
    --num_gpu_cores=2 \
    --data_dir=${GLUE_DIR} \
    --cached_dir=${CURRENT_PWD}/cached_albert_tfrecord \
    --task_name=${TASK} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${MSL} \
    --train_step=${TSP} \
    --warmup_step=${WPSP} \
    --train_batch_size=${BS} \
    --learning_rate=${LR} \
    --albert_config_file=${CURRENT_PWD}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/albert_config.json \
    --init_checkpoint=${CURRENT_PWD}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/model.ckpt-best \
    --vocab_file=./30k-clean.vocab \
    --spm_model_file=./30k-clean.model \
    --save_checkpoints_steps=100 \
    2>&1 | tee ${OUTPUT_DIR}/${TASK}_${ALBERT_DIR}_v${VERSION}.log
