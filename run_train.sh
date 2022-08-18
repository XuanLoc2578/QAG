#!/bin/bash 

python qag_pegasus/run_train.py \
    --model_name_or_path google/pegasus-xsum \
    --cache_dir mounts/models/pegasus-xsum \
    --output_dir mounts/models/gag_pegasus_mrl_model \
    --train_file mounts/data/MCQ_Squad_MRL.csv \
    --source_max_token_len 256 \
    --target_max_token_len 64 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --save_steps 100000000 \
    --logging_steps 100 \
    --save_total_limit 1 \
    --warmup_steps=150 \
    --weight_decay=0.1 \
    --learning_rate=0.00005
