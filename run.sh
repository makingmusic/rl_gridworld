#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up RL Grid World Environment...${NC}"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create and activate virtual environment
echo -e "${BLUE}Creating Python virtual environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
echo -e "${BLUE}Installing required packages...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}Setup complete!${NC}"
echo -e "\nTo run the 1D Grid World example, use:"
echo -e "${BLUE}python 1dworld.py${NC}"
echo -e "\nTo run the 2D Grid World example, use:"
echo -e "${BLUE}python 2dworld.py${NC}"
echo -e "\nNote: The virtual environment is now activated. To deactivate it later, simply type 'deactivate'" 