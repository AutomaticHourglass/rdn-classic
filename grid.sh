#!/usr/bin/env bash
# ------------------------------------------------------------
#  Grid‑search runner for `python -m scripts.base_train`
#
#  • All hyper‑parameters that have more than one value are
#    reflected in the `model_tag` (e.g. _D6_H4_R32).
#  • Heads and KV‑heads are forced to be equal.
#
# ------------------------------------------------------------

set -euo pipefail   # abort on error / unset vars

# ── 1️⃣  Define the search space ---------------------------------
depths=(6 8 4)           # model depth
n_heads_list=(6 8 4)     # number of heads (KV‑heads = heads)
max_seq_lens=(1024)      # max sequence length
device_batch_sizes=(32)  # batch size per device

# These are scalars, so they don't generate a tag
sample_every=100         # how often to sample
eval_every=100           # how often to evaluate

n_streams_list=(1)       # number of streams (GPUs)
aspect_ratios=(128)    # aspect ratio

# ── 2️⃣  Create a log directory ----------------------------------
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── 3️⃣  Nested loops – one for each hyper‑parameter -----------
for depth in "${depths[@]}"; do
    for n_heads in "${n_heads_list[@]}"; do
        # KV‑heads are forced to equal heads
        n_kv_heads=$n_heads

        for max_seq_len in "${max_seq_lens[@]}"; do
            for device_batch_size in "${device_batch_sizes[@]}"; do
                for n_streams in "${n_streams_list[@]}"; do
                    for aspect_ratio in "${aspect_ratios[@]}"; do

                        # ---- 3.1 Build a unique job name ----------------
                        #   model_tag contains only parameters that vary.
                        tag=""

                        [[ ${#depths[@]} -gt 1 ]] && tag+="_D${depth}"
                        [[ ${#n_heads_list[@]} -gt 1 ]] && tag+="_H${n_heads}"
                        [[ ${#max_seq_lens[@]} -gt 1 ]] && tag+="_S${max_seq_len}"
                        [[ ${#device_batch_sizes[@]} -gt 1 ]] && tag+="_B${device_batch_size}"
                        [[ ${#n_streams_list[@]} -gt 1 ]] && tag+="_G${n_streams}"
                        [[ ${#aspect_ratios[@]} -gt 1 ]] && tag+="_R${aspect_ratio}"

                        # Remove leading underscore if present
                        model_tag="${tag#_}"
                        job_name="${model_tag}"
                        log_file="${LOG_DIR}/${job_name}.log"

                        echo "Running job: $job_name"

                        # ---- 3.2 Execute the training command -------------
                        python -m scripts.base_train \
                            --depth="$depth" \
                            --n_heads="$n_heads" \
                            --n_kv_heads="$n_kv_heads" \
                            --max_seq_len="$max_seq_len" \
                            --device_batch_size="$device_batch_size" \
                            --sample_every="$sample_every" \
                            --eval_every="$eval_every" \
                            --n_streams="$n_streams" \
                            --aspect_ratio="$aspect_ratio" \
                            --model_tag="$job_name" \
                            --num_iterations=3000 2>&1

                    done   # aspect_ratio
                done     # n_streams
            done         # device_batch_size
        done             # max_seq_len
    done                 # n_heads
done                     # depth

echo "All grid‑search jobs finished. Logs are in $LOG_DIR/"
