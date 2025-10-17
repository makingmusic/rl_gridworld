#!/bin/bash

# Deactivate any existing virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating existing virtual environment: $VIRTUAL_ENV"
    unset VIRTUAL_ENV
    unset PYTHONPATH
    export PATH=$(echo $PATH | sed 's|[^:]*myenv[^:]*:||g')
fi

# Set custom virtual environment directory
export UV_PROJECT_ENVIRONMENT=myenv1

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up RL Grid World Environment with uv...${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv is not installed. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Initialize uv project if not already done
if [ ! -f "pyproject.toml" ]; then
    echo -e "${BLUE}Initializing uv project...${NC}"
    uv init --no-readme
fi

# Sync dependencies using uv
echo -e "${BLUE}Installing required packages with uv...${NC}"
uv sync

# Check if wandb is installed
if ! uv run --active python -c "import wandb" &> /dev/null; then
    echo -e "${YELLOW}wandb is not installed. Installing wandb...${NC}"
    uv add wandb
fi

# Check if wandb is logged in
if ! uv run --active wandb status | grep -q 'You are logged in as'; then
    echo -e "${YELLOW}wandb is not logged in. Please run 'uv run wandb login' to enable experiment tracking.${NC}"
    echo -e "${YELLOW}You can get your API key from https://wandb.ai/authorize${NC}"
    echo -e "${YELLOW}Note: You can still run the code without wandb, but experiment tracking will be disabled.${NC}"
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "\nTo run the Tabular Q-Learning example, use:"
echo -e "${BLUE}uv run python main.py${NC}"
echo -e "\nTo run the Deep Q-Network (DQN) example, use:"
echo -e "${BLUE}uv run python main_nn.py${NC}"
echo -e "\nNote: uv automatically manages the virtual environment. No need to activate/deactivate manually!" 