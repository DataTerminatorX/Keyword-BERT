#!/bin/bash
# -*- coding: utf-8 -*-

PRE_TRAINED_DIR="pre_trained"
DATA_DIR="data"
OUTOUT_DIR="outputs"
GPU_ID="4"

time_stamp=$(date +"%G-%m-%d - %H-%M")
echo "#########################################"
echo ">>>>> Bert running log @$time_stamp <<<<<"
echo "#########################################"

python run_classifier.py \
  --task_name=Baike \
  --do_train=False \
  --do_eval=False \
  --do_predict=True \
  --do_train_and_eval=False \
  --save_checkpoints_steps=100 \
  --keep_checkpoint_max=3 \
  --num_train_epochs=50 \
  --max_seq_length=128 \
  --train_batch_size=1 \
  --data_dir=$DATA_DIR \
  --vocab_file=$PRE_TRAINED_DIR/vocab.txt \
  --bert_config_file=$PRE_TRAINED_DIR/bert_config_6_layer.json \
  --init_checkpoint=$PRE_TRAINED_DIR/bert_model.ckpt \
  --learning_rate=2e-5 \
  --output_dir=$OUTOUT_DIR \
  --gpu_device_id=$GPU_ID

