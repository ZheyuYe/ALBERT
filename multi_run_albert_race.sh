#!/bin/bash
set -e
set -x

export TASK=RACE
export ALBERT_DIR=base
export VERSION=2
export CURRENT_PWD=/home/ubuntu
export NUM_GPUS=8


export BS=4
export EBS=4
GBS=$(($BS * $NUM_GPUS))
export MSL=512
export MQL=128
export LR=2e-05
export ALBERT_DP=0
export WPSP=1000
export TSP=12000

export RACE_DIR=${CURRENT_PWD}/race_data
export OUTPUT_DIR=${CURRENT_PWD}/albert_output/${TASK}_${ALBERT_DIR}_v${VERSION}_${GBS}_${LR}

if [ ! -d $OUTPUT_DIR  ];then
  mkdir $OUTPUT_DIR
else
  echo $OUTPUT_DIR dir exist
fi

pip3 install numpy
pip3 install -r requirements.txt

set +x

sudo python3 -m albert.run_multigpus_race \
    --do_train=True \
    --do_eval=True \
    --strategy_type=mirror \
    --num_gpu_cores=${NUM_GPUS} \
    --train_feature_file=${CURRENT_PWD}/cached_albert_tfrecord/${TASK}_train.tf_record \
    --eval_feature_file=${CURRENT_PWD}/cached_albert_tfrecord/${TASK}_eval.tf_record \
    --data_dir=${RACE_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${MSL} \
    --max_qa_length=${MQL} \
    --train_step=${TSP} \
    --warmup_step=${WPSP} \
    --train_batch_size=${BS} \
    --eval_batch_size=${EBS} \
    --learning_rate=${LR} \
    --albert_dropout_prob=${ALBERT_DP} \
    --albert_config_file=${CURRENT_PWD}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/albert_config.json \
    --init_checkpoint=${CURRENT_PWD}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/model.ckpt-best \
    --vocab_file=./30k-clean.vocab \
    --spm_model_file=./30k-clean.model \
    --save_checkpoints_steps=250 \
    2>&1 | sudo tee ${OUTPUT_DIR}/${TASK}_${ALBERT_DIR}_v${VERSION}.log
