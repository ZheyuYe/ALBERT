#!/bin/bash
set -e
set -x

export TASK=SQuAD
export VERSION=1
export ALBERT_DIR=base
export SQUAD_VERSION=1.1
export STORAGE_BUCKET=gs://zheyu-albert

export CURRENT_PWD=/content
export SQUAD_DIR=${CURRENT_PWD}/SQuAD_data
export OUTPUT_DIR=${STORAGE_BUCKET}/albert_output/${TASK}${SQUAD_VERSION}_${ALBERT_DIR}_v${VERSION}

export BS=8
export LR=5e-05
export EPOCH=3.0

pip3 install numpy
pip3 install -r requirements.txt

set +x

sudo CUDA_VISIBLE_DEVICES=2 \
sudo python3 -m albert.run_squad_v${SQUAD_VERSION:0:1} \
    --do_train=True \
    --do_predict=True \
    --train_feature_file=${CURRENT_PWD}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION}_train.tfrecord \
    --predict_feature_file=${CURRENT_PWD}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION}_dev.tfrecord \
    --predict_feature_left_file=${CURRENT_PWD}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION}_dev_left \
    --train_file=${SQUAD_DIR}/train-v${SQUAD_VERSION}.json  \
    --predict_file=${SQUAD_DIR}/dev-v${SQUAD_VERSION}.json  \
    --output_dir=${OUTPUT_DIR} \
    --num_train_epochs=${EPOCH} \
    --train_batch_size=${BS} \
    --learning_rate=${LR} \
    --albert_config_file=${CURRENT_PWD}/albert_model/${ALBERT_DIR}_v${VERSION}/albert_config.json \
    --vocab_file=${CURRENT_PWD}/albert_model/${ALBERT_DIR}_v${VERSION}/30k-clean.vocab \
    --spm_model_file=${CURRENT_PWD}/albert_model/${ALBERT_DIR}_v${VERSION}/30k-clean.model \
    --gcs_json_file=/content/albert-a1fd39cb6425.json \
    2>&1 | tee ${CURRENT_PWD}/test.log
