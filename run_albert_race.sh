#!/bin/bash
set -e
set -x

export TASK=RACE
export VERSION=2
export ALBERT_DIR=base

export CURRENT_PWD=/content
export STORAGE_BUCKET=gs://zheyu-albert

export BS=32
export MSL=512
export MQL=128
export LR=2e-05
export WPSP=1000
export TSP=12000

export RACE_DIR=${CURRENT_PWD}/race_data
export OUTPUT_DIR=${STORAGE_BUCKET}/albert_output/${TASK}_${ALBERT_DIR}_v${VERSION}_${BS}_${LR}

pip3 install numpy
pip3 install -r requirements.txt

set +x

sudo python3 -m albert.run_race \
    --do_train=True \
    --do_eval=True \
    --use_tpu=True \
    --tpu_name=grpc://10.77.128.226:8470 \
    --num_tpu_cores=8 \
    --train_feature_file=${STORAGE_BUCKET}/cached_albert_tfrecord/${TASK}_train.tf_record \
    --eval_feature_file=${STORAGE_BUCKET}/cached_albert_tfrecord/${TASK}_eval.tf_record \
    --data_dir=${RACE_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --max_seq_length=${MSL} \
    --max_qa_length=${MQL} \
    --train_step=${TSP} \
    --warmup_step=${WPSP} \
    --train_batch_size=${BS} \
    --learning_rate=${LR} \
    --albert_config_file=${STORAGE_BUCKET}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/albert_config.json \
    --init_checkpoint=${STORAGE_BUCKET}/pretrained_model/albert_${ALBERT_DIR}_v${VERSION}/model.ckpt-best \
    --vocab_file=./30k-clean.vocab \
    --spm_model_file=./30k-clean.model \
    --save_checkpoints_steps=250 \
    2>&1 | sudo tee ${OUTPUT_DIR}/${TASK}_${ALBERT_DIR}_v${VERSION}.log
