#!/bin/bash
# End-to-End Test Suite
# Tests all components with tiny iterations for fast validation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_DIR="test_runs"
FAILED_TESTS=0
PASSED_TESTS=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}End-to-End Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create test directory
mkdir -p $TEST_DIR
rm -rf $TEST_DIR/*

# Helper function to run tests
run_test() {
    local test_name=$1
    local test_cmd=$2

    echo -e "${YELLOW}Testing: $test_name${NC}"
    if eval "$test_cmd" >> "$TEST_DIR/e2e_test.log" 2>&1; then
        echo -e "${GREEN}✓ PASS: $test_name${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL: $test_name${NC}"
        echo -e "${RED}  Log: $TEST_DIR/e2e_test.log${NC}"
        tail -20 "$TEST_DIR/e2e_test.log"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# ============================================================================
# Test 1: Generator Tests
# ============================================================================
echo -e "\n${BLUE}=== Phase 1: Generator Tests ===${NC}"

run_test "calculator_generator" "python -c '
from generator.generator_calc import generate_random_problem
for i in range(10):
    p = generate_random_problem()
    assert \"prompt\" in p
    assert \"json\" in p
    assert \"answer\" in p
print(\"Generated 10 calculator problems successfully\")
'"

run_test "tictactoe_generator" "python -c '
from generator.generator_ttt import generate_random_problem
for i in range(10):
    p = generate_random_problem()
    assert \"conversation\" in p
    assert \"final_board\" in p
    assert len(p[\"conversation\"]) >= 2
print(\"Generated 10 tic-tac-toe games successfully\")
'"

run_test "unified_generator" "python -c '
from generator.unified import generate_random_problem, configure_generators

# Test calculator
configure_generators({\"calculator\": 1.0, \"tic-tac-toe\": 0.0})
for i in range(5):
    p = generate_random_problem()
    assert \"prompt\" in p
    assert \"json\" in p

# Test tic-tac-toe
configure_generators({\"calculator\": 0.0, \"tic-tac-toe\": 1.0})
for i in range(5):
    p = generate_random_problem()
    assert \"prompt\" in p
    assert \"json\" in p

print(\"Unified generator working correctly\")
'"

# ============================================================================
# Test 2: Tokenizer Tests
# ============================================================================
echo -e "\n${BLUE}=== Phase 2: Tokenizer Tests ===${NC}"

run_test "tokenizer_basic" "python -c '
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()

# Test single string
text = \"Hello world\"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
assert isinstance(tokens, list)
assert len(tokens) > 0
print(f\"Single: {text} -> {len(tokens)} tokens\")

# Test list of strings
texts = [\"Hello\", \"world\"]
token_lists = tokenizer.encode(texts)
assert isinstance(token_lists, list)
assert len(token_lists) == 2
print(f\"List: {len(texts)} texts -> {len(token_lists)} token lists\")

# Test with prepend
tokens_with_bos = tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())
assert tokens_with_bos[0] == tokenizer.get_bos_token_id()
print(\"Tokenizer tests passed\")
'"

# ============================================================================
# Test 3: Model Creation Tests
# ============================================================================
echo -e "\n${BLUE}=== Phase 3: Model Tests ===${NC}"

run_test "model_creation" "python -c '
import torch
from nanochat.gpt import GPT, GPTConfig

config = GPTConfig(
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    sequence_len=128,
    vocab_size=320,
    n_streams=1
)

model = GPT(config)
print(f\"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\")

# Test forward pass
B, T = 2, 128
x = torch.randint(0, 320, (B, T))
y = torch.randint(0, 320, (B, T))

loss = model(x, y)
print(f\"Forward pass: loss = {loss.item():.4f}\")
assert not torch.isnan(loss)
assert loss.item() > 0

print(\"Model tests passed\")
'"

run_test "model_mhc" "python -c '
import torch
from nanochat.gpt import GPT, GPTConfig

config = GPTConfig(
    n_layer=2,
    n_embd=64,
    n_head=2,
    n_kv_head=2,
    sequence_len=128,
    vocab_size=320,
    n_streams=4  # Enable mHC
)

model = GPT(config)
print(f\"mHC Model created with {sum(p.numel() for p in model.parameters()):,} parameters\")

# Test forward pass
B, T = 2, 32
x = torch.randint(0, 320, (B, T))
y = torch.randint(0, 320, (B, T))

loss = model(x, y)
print(f\"mHC Forward pass: loss = {loss.item():.4f}\")
assert not torch.isnan(loss)

print(\"mHC model tests passed\")
'"

# ============================================================================
# Test 4: Training Tests (All Agents)
# ============================================================================
echo -e "\n${BLUE}=== Phase 4: Training Tests ===${NC}"

run_test "train_calculator" "python -m scripts.base_train \
    --agent=calculator \
    --depth=2 \
    --n_heads=2 \
    --n_kv_heads=2 \
    --max_seq_len=128 \
    --device_batch_size=2 \
    --total_batch_size=256 \
    --num_iterations=5 \
    --eval_every=5 \
    --core_metric_every=-1 \
    --core_metric_max_per_task=-1 \
    --sample_every=5 \
    --save_every=5 \
    --run=dummy"

run_test "train_tictactoe" "python -m scripts.base_train \
    --agent=tictactoe \
    --depth=2 \
    --n_heads=2 \
    --n_kv_heads=2 \
    --max_seq_len=256 \
    --device_batch_size=2 \
    --total_batch_size=8192 \
    --num_iterations=5 \
    --eval_every=-1 \
    --core_metric_every=-1 \
    --sample_every=-1 \
    --save_every=-1 \
    --run=dummy"

run_test "train_rdn_with_core" "python -m scripts.base_train \
    --agent=rdn \
    --depth=2 \
    --n_heads=2 \
    --n_kv_heads=2 \
    --max_seq_len=256 \
    --device_batch_size=2 \
    --total_batch_size=512 \
    --num_iterations=5 \
    --eval_every=5 \
    --core_metric_every=5 \
    --core_metric_max_per_task=32 \
    --sample_every=5 \
    --save_every=5 \
    --run=dummy"

# ============================================================================
# Test 5: Evaluation Tests
# ============================================================================
echo -e "\n${BLUE}=== Phase 5: Evaluation Tests ===${NC}"

run_test "evaluator_calculator" "python -c '
from generator.evaluator import get_evaluator

eval = get_evaluator(\"calculator\")

# Test parsing
prompt, expected = eval.parse_problem(\"Calculate 5 + 3? # Answer: \\\"8.0\\\"\")
assert prompt == \"Calculate 5 + 3?\"
assert expected == \"8.0\"

# Test evaluation
result = eval.evaluate(prompt, expected, \"blah \\\"8.0\\\" blah\")
assert result[\"corr\"] == 1
assert result[\"score\"] == 1

result = eval.evaluate(prompt, expected, \"blah \\\"5.0\\\" blah\")
assert result[\"corr\"] == 0

print(\"Calculator evaluator tests passed\")
'"

run_test "evaluator_tictactoe" "python -c '
from generator.evaluator import get_evaluator

eval = get_evaluator(\"tictactoe\")

# Test parsing
text = \"Tic-tac-toe. I play center.<|python_start|>board[1][1]=X<|python_end|>\"
prompt, expected = eval.parse_problem(text)
assert prompt is not None
assert expected is not None

# Test evaluation
good_pred = \"<|python_start|>board[1][1]=X\\nprint(board)<|python_end|><|output_start|>board<|output_end|>\"
result = eval.evaluate(prompt, expected, good_pred)
assert result[\"score\"] > 0.5

print(\"Tic-tac-toe evaluator tests passed\")
'"

# ============================================================================
# Test 6: Inference Tests
# ============================================================================
echo -e "\n${BLUE}=== Phase 6: Inference Tests ===${NC}"

run_test "inference_engine" "python -c '
import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine

# Create tiny model
config = GPTConfig(
    n_layer=2,
    n_embd=64,
    n_head=2,
    n_kv_head=2,
    sequence_len=128,
    vocab_size=320,
    n_streams=1
)
model = GPT(config)
model.eval()

tokenizer = get_tokenizer()
engine = Engine(model, tokenizer)

# Test generation
prompt = \"Hello\"
tokens = tokenizer.encode(prompt)

generated = []
with torch.no_grad():
    for token_col, _ in engine.generate_batched([tokens], max_tokens=10, temperature=0):
        generated.append(token_col[0])

assert len(generated) == 10
print(f\"Generated {len(generated)} tokens\")
print(\"Inference engine tests passed\")
'"

# ============================================================================
# Test 7: Dataloader Tests
# ============================================================================
echo -e "\n${BLUE}=== Phase 7: Dataloader Tests ===${NC}"

run_test "dataloader_generator" "python -c '
import torch
from nanochat.dataloader import tokenizing_distributed_data_loader

# Test generator mode
loader = tokenizing_distributed_data_loader(
    B=2,
    T=32,
    split=\"train\",
    device=\"cpu\",
    use_generator=True
)

x, y = next(loader)
assert x.shape == (2, 32)
assert y.shape == (2, 32)
assert x.dtype == torch.long

print(f\"Generator dataloader: x.shape={x.shape}, y.shape={y.shape}\")
print(\"Dataloader tests passed\")
'"

# ============================================================================
# Summary
# ============================================================================
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Check logs in $TEST_DIR/${NC}"
    exit 1
fi
