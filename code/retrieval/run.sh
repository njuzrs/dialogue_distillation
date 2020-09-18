#!/usr/bin/env bash

## for training 
python run_distilling.py \
  --task_name buy_data \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $data_dir \
  --teacher_bert_model $teacher \
  --student_bert_model $student \
  --max_seq_length 64 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --output_dir $output \
  --temperature 1 --alpha 0.5
  
## for testing
python run_ranker_test.py \
  --task_name bug_data \
  --do_eval \
  --do_lower_case \
  --data_dir $data_dir \
  --bert_model $model \
  --max_seq_length 64 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --output_dir $output
