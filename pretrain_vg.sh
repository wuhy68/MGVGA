torchrun --nproc_per_node=8 --master_port=4619 pretrain_vg.py \
    --root_path /OPENABC2/design/ \
    --data_path /data/pretrain/train.csv \
    --pyg_path /datasets/vgalign_dataset/AIG/pyg/ \
    --verilog_path /data/align/verilog_emb_gteqwen7b.npy \
    --output_dir /lcm/mgvga/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --logging_strategy "steps" \
    --save_steps 1000 \
    --logging_steps 10 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --warmup_ratio 0.0 \