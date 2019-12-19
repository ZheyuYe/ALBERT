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

export BS=48
export LR=5e-05
export EPOCH=3.0

pip3 install numpy
pip3 install -r requirements.txt

set +x

sudo python3 -m albert.run_squad_v${SQUAD_VERSION:0:1} \
    --do_train=True \
    --do_predict=True \
    --use_tpu=True \
    --tpu_name=grpc://10.81.144.18:8470 \
    --num_tpu_cores=8 \
    --train_feature_file=${STORAGE_BUCKET}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION}_train.tf_record \
    --predict_feature_file=${STORAGE_BUCKET}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION}_dev.tf_record \
    --predict_feature_left_file=${STORAGE_BUCKET}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION}_dev_left \
    --train_file=${SQUAD_DIR}/train-v${SQUAD_VERSION}.json  \
    --predict_file=${SQUAD_DIR}/dev-v${SQUAD_VERSION}.json  \
    --output_dir=${OUTPUT_DIR} \
    --num_train_epochs=${EPOCH} \
    --train_batch_size=${BS} \
    --learning_rate=${LR} \
    --albert_config_file=${CURRENT_PWD}/albert_model/${ALBERT_DIR}_v${VERSION}/albert_config.json \
    --vocab_file=${CURRENT_PWD}/albert_model/${ALBERT_DIR}_v${VERSION}/30k-clean.vocab \
    --spm_model_file=${CURRENT_PWD}/albert_model/${ALBERT_DIR}_v${VERSION}/30k-clean.model \
    2>&1 | tee ${CURRENT_PWD}/test_warmup.log
