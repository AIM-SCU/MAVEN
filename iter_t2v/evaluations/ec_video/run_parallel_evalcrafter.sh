#!/bin/bash

# Script to run EvalCrafter evaluations in parallel using tmux
#
# Usage:
#   bash /workspace/t2v_secret/iter_t2v/evaluations/ec_video/run_parallel_evalcrafter.sh
#
# Configure these variables:
NUM_GPUS=10       # Number of GPUs (also equals number of partitions)
MODE="sequential"       # Evaluation mode (base, test, etc.)

# Advanced settings (usually don't need to change)
SESSION_PREFIX="evalcrafter"

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EvalCrafter Parallel Evaluation Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Number of GPUs/Partitions: ${YELLOW}${NUM_GPUS}${NC}"
echo -e "Mode: ${YELLOW}${MODE}${NC}"
# Generate partition list
PARTITION_LIST=""
for i in $(seq 1 $NUM_GPUS); do
    PARTITION_LIST="${PARTITION_LIST}${NUM_GPUS}${i} "
done
echo -e "Partitions: ${YELLOW}${PARTITION_LIST}${NC}"
echo ""

# Install tmux if not available
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}tmux not found. Installing...${NC}"
    apt update -qq
    apt install -y tmux
    echo -e "${GREEN}tmux installed successfully!${NC}"
else
    echo -e "${GREEN}tmux is already installed.${NC}"
fi

# Kill existing sessions with the same prefix if they exist
echo -e "\n${YELLOW}Cleaning up existing sessions...${NC}"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    SESSION_NAME="${SESSION_PREFIX}_gpu${i}"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
        echo -e "  Killed existing session: ${SESSION_NAME}"
    fi
done

# Create tmux sessions for each GPU
echo -e "\n${GREEN}Creating tmux sessions...${NC}"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$i
    # Partition format: concatenation of total_partitions and partition_index (1-indexed)
    # Examples: 21 (2 total, partition 1), 101 (10 total, partition 1), 1010 (10 total, partition 10)
    PARTITION="${NUM_GPUS}$((i + 1))"
    SESSION_NAME="${SESSION_PREFIX}_gpu${GPU_ID}"

    # Create detached tmux session
    tmux new-session -d -s "$SESSION_NAME"

    # Set working directory
    tmux send-keys -t "$SESSION_NAME" "cd /workspace/t2v_secret/iter_t2v" C-m

    # Activate conda environment if needed (uncomment if you have a specific env)
    # tmux send-keys -t "$SESSION_NAME" "conda activate your_env_name" C-m

    # Set HuggingFace environment variables
    tmux send-keys -t "$SESSION_NAME" "export HF_HOME=/workspace/hf_home" C-m
    tmux send-keys -t "$SESSION_NAME" "echo \$HF_HOME" C-m
    tmux send-keys -t "$SESSION_NAME" "export HF_HUB_ENABLE_HF_TRANSFER=1" C-m
    tmux send-keys -t "$SESSION_NAME" "echo \$HF_HUB_ENABLE_HF_TRANSFER" C-m

    # Run the evaluation command
    CMD="python evaluations/ec_video/run_evalcrafter.py --mode ${MODE} --partition ${PARTITION} --gpu ${GPU_ID}"
    echo -e "  ${GREEN}✓${NC} Session: ${SESSION_NAME} | GPU: ${GPU_ID} | Partition: ${PARTITION}"
    echo -e "    Command: ${CMD}"
    tmux send-keys -t "$SESSION_NAME" "$CMD" C-m
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All sessions started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Useful tmux commands:${NC}"
echo -e "  • List sessions:        ${GREEN}tmux ls${NC}"
echo -e "  • Attach to session:    ${GREEN}tmux attach -t ${SESSION_PREFIX}_gpu0${NC}"
echo -e "  • Detach from session:  ${GREEN}Ctrl+b then d${NC}"
echo -e "  • Kill all sessions:    ${GREEN}tmux kill-server${NC}"
echo ""
echo -e "${YELLOW}Monitor individual sessions:${NC}"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo -e "  • GPU ${i}: ${GREEN}tmux attach -t ${SESSION_PREFIX}_gpu${i}${NC}"
done
echo ""

# Optional: Show active sessions
echo -e "${GREEN}Active tmux sessions:${NC}"
tmux ls 2>/dev/null || echo "No active sessions"
