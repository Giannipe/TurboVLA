#!/bin/bash
# Source from train_libero.sbatch. Numeric/model settings are upstream b29ab14.
# paper256 changes ONLY per-device batch to the pre-correction value (16).
case "${recipe:?}" in
  upstream128) per_device_batch=8 ;;
  paper256) per_device_batch=16 ;;
  *) echo "Unknown recipe: $recipe (use upstream128 or paper256)" >&2; return 2 ;;
esac
datasets="${TURBOVLA_STORE}/datasets/libero"
train_args=(
  --dataset_dir "$datasets/libero_10_no_noops/1.0.0"
  --dataset_dirs "$datasets/libero_10_no_noops/1.0.0,$datasets/libero_goal_no_noops/1.0.0,$datasets/libero_object_no_noops/1.0.0,$datasets/libero_spatial_no_noops/1.0.0"
  --dataset_split train
  --stats_path "$source_root/experiments/libero/configs/libero_all4_stats.json"
  --stats_key libero_all4_no_noops --normalize_binary_gripper auto
  --dinov3_path "$TURBOVLA_STORE/models/dinov3-vitb16"
  --bert_path "$TURBOVLA_STORE/models/bert-base-uncased"
  --pretrained_init_ckpt "$TURBOVLA_STORE/models/groundingdino/groundingdino_swint_ogc.pth"
  --no_allow_hf_download
  --require_feature_enhancer_preload --load_text_projection_from_init --require_text_proj_preload
  --no_freeze_backbones --freeze_text_encoder
  --batch_size "$per_device_batch" --grad_accum_steps 4
  --lr 5e-5 --head_lr 5e-5 --dinov3_lr 5e-5
  --weight_decay 1e-10 --head_weight_decay 1e-10 --dinov3_weight_decay 1e-10
  --precision fp32 --dinov3_precision bf16_autocast
  --max_steps "$max_steps" --lr_schedule_steps 80000 --warmup_steps 10000
  --min_lr_ratio 1.0 --max_grad_norm 1.0
  --save_steps 1000 --save_final --log_freq 20
  --num_workers 4 --shuffle_buffer 512 --step_mix_buffer_size 64
  --shuffle_steps_within_episode --expected_image_size 256
  --hidden_dim 256 --nheads 8 --dim_feedforward 2048
  --max_text_len 256 --text_padding_length 21
  --text_layout_path "$source_root/experiments/libero/configs/online_text_layout.json"
  --vla_feature_enhancer_layers 6 --enhancer_inner_dim 1024
  --action_dim 7 --chunk_size 12 --state_dim 8 --num_state_tokens 2
  --text_dropout 0.0 --fusion_dropout 0.0 --fusion_droppath 0.1 --seed 42
  --checkpoint_dir "$run_dir/checkpoints" --checkpoint_prefix turbovla_step
  --resume_mode "$resume_mode"
)
