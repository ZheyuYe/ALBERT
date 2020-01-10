#!/bin/bash
set -e
set -x

export TASK=CoLA
export VERSION=2
export ALBERT_DIR=base

export CURRENT_PWD=/content
export STORAGE_BUCKET=gs://zheyu-albert

export BS=32
export MSL=128
export LR=1e-05
export ALBERT_DP=0
export WPSP=630
export TSP=10672

export GLUE_DIR=${CURRENT_PWD}/glue_data
export OUTPUT_DIR=${STORAGE_BUCKET}/albert_output/${TASK}_${ALBERT_DIR}_v${VERSION}_${BS}_${LR}

pip3 install numpy
pip3 install -r requirements.txt

set +x

sudo python3 -m albert.run_classifier \
    --do_train=True \
    --do_eval=True \
    --use_tpu=True \
    --tpu_name=grpc://10.14.119.162:8470 \
    --num_tpu_cores=8 \
    --data_dir=${GLUE_DIR} \
    --cached_dir=${STORAGE_BUCKET}/cached_albert_tfrecord \
    --task_name=${TASK} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${MSL} \
    --train_step=${TSP} \
    --warmup_step=${WPSP} \
    --train_batch_size=${BS} \
    --learning_rate=${LR} \
    --albert_dropout_prob=${ALBERT_DP} \
    --albert_config_file=${STORAGE_BUCKET}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/albert_config.json \
    --init_checkpoint=${STORAGE_BUCKET}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/model.ckpt-best \
    --vocab_file=./30k-clean.vocab \
    --spm_model_file=./30k-clean.model \
    --save_checkpoints_steps=250 \
    2>&1 | sudo tee ${OUTPUT_DIR}/${TASK}_${ALBERT_DIR}_v${VERSION}.log
