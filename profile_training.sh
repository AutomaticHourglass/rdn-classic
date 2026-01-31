#!/bin/bash
# Training profiler for gpt.py and base_train.py
# Works on Linux (CUDA) and Mac
# Outputs detailed line-by-line timing information

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Training Profiler${NC}"
echo -e "${BLUE}========================================${NC}"

# Parse arguments
PROFILE_MODE=${1:-"quick"}  # quick, detailed, or full
AGENT=${2:-"calculator"}
DEPTH=${3:-4}

echo -e "${GREEN}Profile mode: $PROFILE_MODE${NC}"
echo -e "${GREEN}Agent: $AGENT${NC}"
echo -e "${GREEN}Depth: $DEPTH${NC}"
echo ""

# Create output directory
PROFILE_DIR="profile_results"
mkdir -p $PROFILE_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PREFIX="${PROFILE_DIR}/profile_${AGENT}_d${DEPTH}_${TIMESTAMP}"

# Check if py-spy is installed
if ! command -v py-spy &> /dev/null; then
    echo -e "${YELLOW}py-spy not found. Installing...${NC}"
    pip install py-spy
fi

# Check if line_profiler is installed (for detailed mode)
if [ "$PROFILE_MODE" == "detailed" ] || [ "$PROFILE_MODE" == "full" ]; then
    if ! python -c "import line_profiler" 2>/dev/null; then
        echo -e "${YELLOW}line_profiler not found. Installing...${NC}"
        pip install line_profiler
    fi
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Phase 1: Quick Profiling with py-spy${NC}"
echo -e "${BLUE}========================================${NC}"

# Run training with py-spy (sampling profiler, no code changes needed)
echo -e "${GREEN}Running py-spy profiler...${NC}"
echo "This will run a short training session and capture performance data."
echo ""

# Determine number of iterations based on mode
if [ "$PROFILE_MODE" == "quick" ]; then
    NUM_ITER=10
elif [ "$PROFILE_MODE" == "detailed" ]; then
    NUM_ITER=50
else
    NUM_ITER=100
fi

# Run with py-spy
py-spy record \
    --output "${OUTPUT_PREFIX}_pyspy.txt" \
    --format text \
    --rate 100 \
    --subprocesses \
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

echo ""
echo -e "${GREEN}✓ py-spy profiling complete!${NC}"
echo -e "Output: ${OUTPUT_PREFIX}_pyspy.txt"
echo ""

# Also generate a speedscope format for visualization
echo -e "${GREEN}Generating speedscope format...${NC}"
py-spy record \
    --output "${OUTPUT_PREFIX}_pyspy.speedscope.json" \
    --format speedscope \
    --rate 100 \
    --subprocesses \
    -- python -m scripts.base_train -- \
        --agent=$AGENT \
        --depth=$DEPTH \
        --max_seq_len=512 \
        --device_batch_size=4 \
        --num_iterations=$NUM_ITER \
        --eval_every=-1 \
        --core_metric_every=-1 \
        --sample_every=-1 \
        --save_every=-1 \
    2>/dev/null || true

echo -e "${GREEN}✓ Speedscope format generated (upload to speedscope.app)${NC}"
echo ""

# Detailed profiling with line_profiler
if [ "$PROFILE_MODE" == "detailed" ] || [ "$PROFILE_MODE" == "full" ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Phase 2: Detailed Line Profiling${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Create a profiling script that wraps the training
    PROFILER_SCRIPT="${PROFILE_DIR}/profiler_wrapper.py"

    cat > $PROFILER_SCRIPT << 'PROFILER_EOF'
import sys
import os
from line_profiler import LineProfiler

# Import the modules to profile
sys.path.insert(0, os.getcwd())
from nanochat import gpt
from scripts import base_train

# Key functions to profile in gpt.py
profile_functions = [
    gpt.GPT.forward,
    gpt.Block.forward,
    gpt.CausalSelfAttention.forward,
    gpt.MLP.forward,
]

# Try to add MHCBlock if it exists
try:
    profile_functions.append(gpt.MHCBlock.forward)
except AttributeError:
    pass

# Create profiler
profiler = LineProfiler()
for func in profile_functions:
    profiler.add_function(func)

# Wrap and run
profiler.enable()

# Run base_train
try:
    exec(open('scripts/base_train.py').read())
except SystemExit:
    pass

profiler.disable()

# Print results
profiler.print_stats()
PROFILER_EOF

    echo -e "${GREEN}Running line profiler on key functions...${NC}"
    echo "Profiling: GPT.forward, Block.forward, Attention.forward, MLP.forward"
    echo ""

    python $PROFILER_SCRIPT \
        --agent=$AGENT \
        --depth=$DEPTH \
        --max_seq_len=512 \
        --device_batch_size=4 \
        --num_iterations=20 \
        --eval_every=-1 \
        --core_metric_every=-1 \
        --sample_every=-1 \
        --save_every=-1 \
        2>&1 | tee "${OUTPUT_PREFIX}_lineprof.txt"

    echo ""
    echo -e "${GREEN}✓ Line profiling complete!${NC}"
    echo -e "Output: ${OUTPUT_PREFIX}_lineprof.txt"
    echo ""
fi

# Generate summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Profiling Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Generated files:${NC}"
ls -lh ${OUTPUT_PREFIX}*
echo ""

# Parse py-spy output for top bottlenecks
echo -e "${YELLOW}Top 20 functions by time:${NC}"
echo -e "${YELLOW}------------------------${NC}"
if [ -f "${OUTPUT_PREFIX}_pyspy.txt" ]; then
    head -n 30 "${OUTPUT_PREFIX}_pyspy.txt"
    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Profiling Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Results saved to: ${BLUE}${OUTPUT_PREFIX}*${NC}"
echo ""
echo -e "${YELLOW}How to view results:${NC}"
echo -e "  1. Text output:    cat ${OUTPUT_PREFIX}_pyspy.txt"
echo -e "  2. Line profile:   cat ${OUTPUT_PREFIX}_lineprof.txt"
echo -e "  3. Visualization:  Upload ${OUTPUT_PREFIX}_pyspy.speedscope.json to https://speedscope.app"
echo ""
echo -e "${YELLOW}Tips:${NC}"
echo -e "  - Look for high 'Tot Time' in py-spy output"
echo -e "  - Check 'Hits' and 'Time' columns in line profiler"
echo -e "  - Focus on lines that contribute >5% of total time"
echo ""
