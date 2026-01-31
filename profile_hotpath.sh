#!/bin/bash
# Fast profiler focused on hot path (forward/backward pass)
# Minimal overhead, quick results

set -e

AGENT=${1:-"calculator"}
DEPTH=${2:-4}
NUM_ITER=${3:-20}

echo "========================================="
echo "Hot Path Profiler"
echo "========================================="
echo "Agent: $AGENT | Depth: $DEPTH | Iterations: $NUM_ITER"
echo ""

# Install py-spy if needed
if ! command -v py-spy &> /dev/null; then
    echo "Installing py-spy..."
    pip install py-spy -q
fi

# Create output dir
mkdir -p profile_results
OUTPUT="profile_results/hotpath_${AGENT}_d${DEPTH}_$(date +%H%M%S).txt"

echo "Profiling training loop..."
echo ""

# Run with high sampling rate for accuracy
py-spy record \
    --rate 200 \
    --format raw \
    --output $OUTPUT \
    -- python -m scripts.base_train -- \
        --agent=$AGENT \
        --depth=$DEPTH \
        --max_seq_len=512 \
        --device_batch_size=4 \
        --num_iterations=$NUM_ITER \
        --eval_every=-1 \
        --core_metric_every=-1 \
        --sample_every=-1 \
        --save_every=-1

# Show results
echo ""
cat $OUTPUT

echo ""
echo "========================================="
echo "Results saved: $OUTPUT"
echo "========================================="
