#!/bin/bash
# DreamZero rx2_blackbox training script.
#
# Data layout expected at DATA_ROOT (produced by blackbox2lerobot):
#   <task>/sortie_XXXX_YYYY_train..._encoder_complete/{data,meta,videos}
#   <task>/sortie_XXXX_YYYY_val..._encoder_complete/{data,meta,videos}
#
# Each *train* sub-directory must have been processed by
# scripts/data/convert_rx2_blackbox_to_gear.py first (meta/modality.json,
# meta/embodiment.json, meta/stats.json, meta/relative_stats_dreamzero.json).
# Run scripts/data/prepare_rx2_blackbox.sh once to convert them all.
#
# Resolution note:
#   Source videos are 640x480 (4:3). Pretrained DreamZero-AgiBot was trained at
#   320x176 (16:9, frame_seqlen=880). We resize to 320x240 (4:3) so aspect ratio
#   is preserved -- LoRA has to adapt some positional encoding. If training
#   quality is poor, fall back to 320x176 (aspect stretched, matches pretrained
#   position embeddings exactly).
#
# Usage:
#   DATA_ROOT=/data/nas_ray/dataset/foundation_data/processed/lerobot/rx2_blackbox \
#     bash scripts/train/rx2_blackbox_training.sh

set -euo pipefail
export HYDRA_FULL_ERROR=1

# ============ CONFIGURATION ============
DATA_ROOT=${DATA_ROOT:?"Set DATA_ROOT to the rx2_blackbox root (contains one dir per task)"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_rx2_blackbox_lora"}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-"./checkpoints/DreamZero-AgiBot"}
# =======================================

# Auto-download missing pretrained weights
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not at $WAN_CKPT_DIR, downloading..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl not at $TOKENIZER_DIR, downloading..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
if [ ! -d "$PRETRAINED_MODEL" ]; then
    echo "ERROR: DreamZero-AgiBot pretrained not found at $PRETRAINED_MODEL"
    echo "Download with:"
    echo "  hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir $PRETRAINED_MODEL"
    exit 1
fi

# Discover every *_train*_encoder_complete sub-dataset under DATA_ROOT.
DATASETS=()
for TASK_DIR in "$DATA_ROOT"/*/; do
    [ -d "$TASK_DIR" ] || continue
    for SUB in "$TASK_DIR"*train*_encoder_complete/; do
        [ -d "$SUB" ] || continue
        [ -f "$SUB/meta/modality.json" ] || {
            echo "SKIP (no modality.json, run prepare_rx2_blackbox.sh first): $SUB" >&2
            continue
        }
        DATASETS+=("${SUB%/}")
    done
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: no prepared *_train*_encoder_complete datasets found under $DATA_ROOT" >&2
    echo "Run scripts/data/prepare_rx2_blackbox.sh $DATA_ROOT first." >&2
    exit 1
fi

# Build the Hydra list literal '["/a","/b",...]'
DATASETS_HYDRA="["
for i in "${!DATASETS[@]}"; do
    if [ "$i" -gt 0 ]; then DATASETS_HYDRA+=","; fi
    DATASETS_HYDRA+="\"${DATASETS[$i]}\""
done
DATASETS_HYDRA+="]"

echo "Training on ${#DATASETS[@]} sub-dataset(s):"
for d in "${DATASETS[@]}"; do echo "  - $d"; done
echo

# num_frames=33 and frame_seqlen=1200 match a 320x240 image resized at 8x VAE
# downsample (320/8 * 240/8 = 40 * 30 = 1200 tokens per frame).
torchrun --nproc_per_node "$NUM_GPUS" --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/rx2_blackbox_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=1 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=10000 \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=4 \
    max_steps=100000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=240 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=1200 \
    save_strategy=steps \
    "rx2_blackbox_datasets=$DATASETS_HYDRA" \
    dit_version="$WAN_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR" \
    pretrained_model_path="$PRETRAINED_MODEL" \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
