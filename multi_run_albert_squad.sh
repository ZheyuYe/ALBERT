#!/bin/bash
set -e
set -x

export TASK=SQuAD
export VERSION=2
export ALBERT_DIR=base
export SQUAD_VERSION=2.0
export CURRENT_PWD=/home/ubuntu
export NUM_GPUS=8

export BS=16
export EBS=4
GBS=$(($BS * $NUM_GPUS))
export LR=5e-05
export EPOCH=3.0
export MSL=512

export SQUAD_DIR=${CURRENT_PWD}/SQuAD_data
export OUTPUT_DIR=${CURRENT_PWD}/albert_output/${TASK}${SQUAD_VERSION}_${ALBERT_DIR}_v${VERSION}_${GBS}_${LR}

if [ ! -d $OUTPUT_DIR  ];then
  mkdir $OUTPUT_DIR
else
  echo $OUTPUT_DIR dir exist
fi

pip3 install numpy
pip3 install -r requirements.txt

set +x

sudo python3 -m albert.run_multigpus_squad_v${SQUAD_VERSION:0:1} \
    --do_train=True \
    --do_predict=True \
    --strategy_type=mirror \
    --num_gpu_cores=${NUM_GPUS} \
    --train_feature_file=${CURRENT_PWD}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION}_train_${MSL}.tf_record \
    --predict_feature_file=${CURRENT_PWD}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION}_dev_${MSL}.tf_record \
    --predict_feature_left_file=${CURRENT_PWD}/cached_albert_tfrecord/${TASK}${SQUAD_VERSION}_dev_left_${MSL} \
    --train_file=${SQUAD_DIR}/train-v${SQUAD_VERSION}.json  \
    --predict_file=${SQUAD_DIR}/dev-v${SQUAD_VERSION}.json  \
    --output_dir=${OUTPUT_DIR} \
    --num_train_epochs=${EPOCH} \
    --train_batch_size=${BS} \
    --eval_batch_size=${EBS} \
    --learning_rate=${LR} \
    --max_seq_length=${MSL} \
    --albert_config_file=${CURRENT_PWD}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/albert_config.json \
    --init_checkpoint=${CURRENT_PWD}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/model.ckpt-best \
    --spm_model_file=./30k-clean.model \
    --vocab_file=./30k-clean.vocab \
    --save_checkpoints_steps=250 \
    2>&1 | sudo tee ${OUTPUT_DIR}/${TASK}${SQUAD_VERSION}_${ALBERT_DIR}_v${VERSION}.log
