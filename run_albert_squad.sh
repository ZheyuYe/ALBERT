#!/bin/bash
set -e
set -x

export TASK=SQuAD
export VERSION=1
export ALBERT_DIR=base
export SQUAD_VERSION=1.1

export CURRENT_PWD=/home/ubuntu
export SQUAD_DIR=${CURRENT_PWD}/SQuAD_data
export OUTPUT_DIR=${CURRENT_PWD}/sp_output/${TASK}${SQUAD_VERSION}_${ALBERT_DIR}_v${VERSION}
export LOG_DIR=own
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir $OUTPUT_DIR
fi

export BS=8
export LR=5e-05
export EPOCH=3.0

pip3 install numpy
pip3 install -r requirements.txt

set +x

sudo CUDA_VISIBLE_DEVICES=2 \
sudo python3 -m albert.run_squad_sp \
    --do_train=True \
    --do_predict=True \
    --feature_file=${CURRENT_PWD}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION} \
    --train_file=${SQUAD_DIR}/train-v${SQUAD_VERSION}.json  \
    --predict_file=${SQUAD_DIR}/dev-v${SQUAD_VERSION}.json  \
    --output_dir=${OUTPUT_DIR} \
    --num_train_epochs=${EPOCH} \
    --train_batch_size=${BS} \
    --learning_rate=${LR} \
    --pretrained_model_dir=${CURRENT_PWD}/albert_model/${ALBERT_DIR}_v${VERSION}/ \
    2>&1 | tee ${OUTPUT_DIR}/${LOG_DIR}.log
