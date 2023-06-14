# This is the training script that was used on a lower-end gpu during
# the first training run.

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="signs-set-finetuned"
export TRAIN_DATA="signs-set"
export CAPTION_COLUMN="caption"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --train_data_dir=$TRAIN_DATA \
  --caption_column=$CAPTION_COLUMN \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="Totoro" \
  --seed=1337