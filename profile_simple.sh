#!/bin/bash
# Simple profiler using Python's built-in cProfile (no sudo needed)

set -e

AGENT=${1:-"calculator"}
DEPTH=${2:-4}
NUM_ITER=${3:-10}

echo "========================================="
echo "Simple Profiler (cProfile)"
echo "========================================="
echo "Agent: $AGENT | Depth: $DEPTH | Iterations: $NUM_ITER"
echo ""

mkdir -p profile_results
OUTPUT="profile_results/cprofile_${AGENT}_d${DEPTH}_$(date +%H%M%S)"

echo "Profiling training loop..."
echo ""

# Run with cProfile (no -- separator needed)
python -m cProfile -o "${OUTPUT}.prof" -m scripts.base_train \
    --agent=$AGENT \
    --depth=$DEPTH \
    --n_heads=4 \
    --n_kv_heads=4 \
    --max_seq_len=512 \
    --device_batch_size=4 \
    --num_iterations=$NUM_ITER \
    --eval_every=-1 \
    --core_metric_every=-1 \
    --sample_every=-1 \
    --save_every=-1

echo ""
echo "Converting to text..."

# Convert to readable text
python -c "
import pstats
prof = pstats.Stats('${OUTPUT}.prof')
prof.strip_dirs()

# Save full stats
with open('${OUTPUT}.txt', 'w') as f:
    # Top functions by cumulative time
    f.write('=' * 80 + '\n')
    f.write('TOP 50 FUNCTIONS BY CUMULATIVE TIME\n')
    f.write('=' * 80 + '\n')
    prof.stream = f
    prof.sort_stats('cumulative')
    prof.print_stats(50)

    # Top functions by internal time
    f.write('\n\n')
    f.write('=' * 80 + '\n')
    f.write('TOP 50 FUNCTIONS BY INTERNAL TIME\n')
    f.write('=' * 80 + '\n')
    prof.sort_stats('time')
    prof.print_stats(50)

    # Focus on gpt.py
    f.write('\n\n')
    f.write('=' * 80 + '\n')
    f.write('GPT.PY FUNCTIONS\n')
    f.write('=' * 80 + '\n')
    prof.sort_stats('cumulative')
    prof.print_stats('gpt.py')

print(f'Results saved to: ${OUTPUT}.txt')
"

echo ""
echo "========================================="
echo "Top 30 bottlenecks:"
echo "========================================="
head -n 40 "${OUTPUT}.txt" | tail -n 30

echo ""
echo "========================================="
echo "Full results: ${OUTPUT}.txt"
echo "========================================="
