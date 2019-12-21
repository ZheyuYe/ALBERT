ALBERT
======
Cloned from [Google ALBERT](https://github.com/google-research/ALBERT) which only suitable for CPU, single GPU and TPU. We consider the favour of [bert-multi-gpu](https://github.com/HaoyuHu/bert-multi-gpu) and did slight modificartion to support Multi-GPU fine-tuning on AWS P3.16xlarge.

**Modification Details**

1. reset the the [Estimator](https://github.com/ZheyuYe/ALBERT/blob/master/albert/run_multigpus_classifier.py#L316) and [EstimatorSpec](https://github.com/ZheyuYe/ALBERT/blob/master/albert/classifier_utils.py#L889) cause the oringinal one could ony suitable for single training device.

2. Adapt MirroredStrategy into training progress as [here](https://github.com/ZheyuYe/ALBERT/blob/master/albert/run_multigpus_classifier.py#L253). Notice: the input data is batched by the global batch size, whereas the batch size setting in the parameters of `FLAGS` are local batch size. 

3. Transform the optimizer including AdamW and Lamb in [custom_optimization.py](https://github.com/ZheyuYe/ALBERT/blob/master/albert/custom_optimization.py)

   

#### Data and Evalution scripts

SQuAD

*   [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
*   [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
*   [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)
*   [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
*   [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
*   [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

GLUE

simply use`python3 download_glue_data.py` to download **ALL GLUE TASKS**

#### Simply Fine-tuning

1. simply load and save the pre-trained model by running the bash file `download_pretrained_models.sh`

2. Use `multi_run_albert_glue.sh` to fine-tune ALBERT on GLUE and `multi_run_albert_squad.sh` to fine-tune on SQuAD

3. DON'T forget to set up file path inside bash file.

   ```bash
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
   ```

