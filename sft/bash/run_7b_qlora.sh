export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
OUTPUT_DIR=./output/alpaca-7b
torchrun --nproc_per_node=1 train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/ \
    --dataset alpaca_data_1k.json \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --save_strategy "epoch" \
    --save_steps 2 \
    --save_total_limit 2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --qlora True \