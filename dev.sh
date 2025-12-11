#!/bin/bash
# Development helper script for OKADFA project

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VENV_PATH="/mnt/storage/dev/dev/ortho-ai-research/.venv"

echo -e "${BLUE}OKADFA Development Helper${NC}"
echo "================================"
echo ""

# Function to run with venv python
run_python() {
    $VENV_PATH/bin/python "$@"
}

# Parse command
case "$1" in
    verify)
        echo -e "${GREEN}Verifying installation...${NC}"
        run_python scripts/verify_installation.py
        ;;
    test)
        echo -e "${GREEN}Running tests...${NC}"
        $VENV_PATH/bin/pytest tests/ -v
        ;;
    format)
        echo -e "${GREEN}Formatting code...${NC}"
        $VENV_PATH/bin/black src/ tests/ scripts/
        $VENV_PATH/bin/isort src/ tests/ scripts/
        ;;
    lint)
        echo -e "${GREEN}Linting code...${NC}"
        $VENV_PATH/bin/flake8 src/ tests/ scripts/
        ;;
    train)
        echo -e "${GREEN}Starting training...${NC}"
        run_python scripts/train.py "${@:2}"
        ;;
    shell)
        echo -e "${GREEN}Starting Python shell with OKADFA environment...${NC}"
        $VENV_PATH/bin/python
        ;;
    jupyter)
        echo -e "${GREEN}Starting Jupyter notebook...${NC}"
        $VENV_PATH/bin/jupyter notebook
        ;;
    *)
        echo "Usage: ./dev.sh [command]"
        echo ""
        echo "Commands:"
        echo "  verify    - Verify installation"
        echo "  test      - Run all tests"
        echo "  format    - Format code with black and isort"
        echo "  lint      - Lint code with flake8"
        echo "  train     - Start training (WIP)"
        echo "  shell     - Start Python shell"
        echo "  jupyter   - Start Jupyter notebook"
        echo ""
        echo "Example: ./dev.sh verify"
        ;;
esac
