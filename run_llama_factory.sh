#!/bin/bash

# ---------- user-specific bits ----------
RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then
  echo "Usage: $0 <RUN_NAME>"
  exit 1
fi
CONFIG=examples/train_full/$RUN_NAME.yaml
SAVE_ROOT=/home/jovyan/workspace/saves/qwen2_5_vl-3b/full/$RUN_NAME
CKPT_GLOB="checkpoint-*"
LOGFILE=/home/jovyan/shared/RodDeSc/experiments/logs/$RUN_NAME.out
# ---------------------------------------

echo "==============================================="
echo "Run Name          : $RUN_NAME"
echo "Config File       : $CONFIG"
echo "Save Root         : $SAVE_ROOT"
echo "Log File          : $LOGFILE"
echo "CUDA Devices      : $CUDA_VISIBLE_DEVICES"
echo "-----------------------------------------------"

source activate llama
conda info

# Pick the most recent checkpoint directory (if any)
ckpt_pattern="${SAVE_ROOT}/checkpoint-*"
# Pick the most recent checkpoint directory (if any)
LATEST_CKPT=$(ls -d $ckpt_pattern 2>/dev/null | sort -V | tail -n 1)
SECOND_LATEST_CKPT=$(ls -d  $ckpt_pattern 2>/dev/null | sort -V | head -n 1)

if [[ -n "${LATEST_CKPT}" && -d "${LATEST_CKPT}" ]]; then
    echo "🟢 Found checkpoint -> ${LATEST_CKPT}.  Resuming training."
    EXTRA_ARGS="resume_from_checkpoint=${LATEST_CKPT}"
    EXTRA_ARGS_BACKUP="resume_from_checkpoint=${SECOND_LATEST_CKPT}"
else
    echo "🟡 No previous checkpoint found.  Starting from scratch."
    EXTRA_ARGS=""
fi

# NOTE: put overrides *after* the yaml file so they win.
llamafactory-cli train "${CONFIG}" ${EXTRA_ARGS} >> "$LOGFILE" 2>&1
# This will run only if the first run fails
llamafactory-cli train "${CONFIG}" ${EXTRA_ARGS_BACKUP} >> "$LOGFILE" 2>&1
