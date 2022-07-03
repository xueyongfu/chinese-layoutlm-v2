
# RUN

- 训练 v1
`python run_xfun_ser.py --model_name_or_path ../DATA/pretrained-models/layoutxlm-base --output_dir ../DATA/huawei-invoice/models --logging_dir ../DATA/huawei-invoice/runs --do_train --do_eval --lang zh --num_train_epochs 100 --warmup_ratio 0.1 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --save_steps 300 --logging_steps 300 --evaluation_strategy steps --eval_steps 300`

- 训练 v2