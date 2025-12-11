#!/bin/bash
#
# OKADFA Experiment Runner
# Streamlined script to run common experiments
#
# Usage:
#   ./scripts/run_experiment.sh [experiment_name]
#
# Available experiments:
#   quick       - Quick test (100 steps, ~3 min)
#   extended    - Extended training (1000 steps, ~30 min)
#   benchmark   - Benchmark comparison (~20 min)
#   full        - Full training (10000 steps, ~5 hours)
#   analyze     - Analyze latest results
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓${NC} Virtual environment activated"
else
    echo -e "${RED}✗${NC} Virtual environment not found. Run: python3 -m venv .venv"
    exit 1
fi

# Check if PyTorch is available
python -c "import torch" 2>/dev/null || {
    echo -e "${RED}✗${NC} PyTorch not found. Run: pip install -e ."
    exit 1
}

# Function to print header
print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Function to run with timestamp
run_with_timestamp() {
    local name=$1
    shift
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="logs/${name}_${timestamp}.log"
    
    mkdir -p logs
    
    echo -e "${YELLOW}▶${NC} Running: $name"
    echo -e "${YELLOW}▶${NC} Log file: $log_file"
    echo ""
    
    # Run command and tee to log file
    "$@" 2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓${NC} $name completed successfully"
        echo -e "${GREEN}✓${NC} Results saved to: $log_file"
    else
        echo ""
        echo -e "${RED}✗${NC} $name failed with exit code $exit_code"
        echo -e "${RED}✗${NC} Check log file: $log_file"
        exit $exit_code
    fi
}

# Parse experiment name
EXPERIMENT=${1:-"help"}

case "$EXPERIMENT" in
    quick)
        print_header "OKADFA Quick Test (100 steps, ~3 minutes)"
        run_with_timestamp "quick_test" \
            python scripts/train_wikitext.py \
                --quick_test \
                --device cpu
        
        echo ""
        echo -e "${GREEN}▶${NC} Analyzing results..."
        python scripts/analyze_results.py \
            --log_file logs/quick_test_*.log \
            --output_dir analysis/quick_test
        ;;
    
    extended)
        print_header "OKADFA Extended Training (1000 steps, ~30 minutes)"
        run_with_timestamp "extended_training" \
            python scripts/train_wikitext.py \
                --max_steps 1000 \
                --batch_size 4 \
                --eval_interval 100 \
                --save_interval 200 \
                --checkpoint_dir checkpoints_extended \
                --d_model 256 \
                --num_layers 2 \
                --num_heads 4 \
                --device cpu
        
        echo ""
        echo -e "${GREEN}▶${NC} Analyzing results..."
        python scripts/analyze_results.py \
            --log_file logs/extended_training_*.log \
            --output_dir analysis/extended
        ;;
    
    benchmark)
        print_header "OKADFA Benchmark Comparison (~20 minutes)"
        run_with_timestamp "benchmark" \
            python scripts/benchmark_okadfa.py \
                --quick_test \
                --device cpu
        ;;
    
    full)
        print_header "OKADFA Full Training (10000 steps, ~5 hours)"
        echo -e "${YELLOW}⚠${NC}  This will take approximately 5 hours on CPU"
        echo -e "${YELLOW}⚠${NC}  Consider using GPU with --device cuda"
        echo ""
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            run_with_timestamp "full_training" \
                python scripts/train_wikitext.py \
                    --max_steps 10000 \
                    --batch_size 4 \
                    --eval_interval 500 \
                    --save_interval 1000 \
                    --checkpoint_dir checkpoints_full \
                    --device cpu
            
            echo ""
            echo -e "${GREEN}▶${NC} Analyzing results..."
            python scripts/analyze_results.py \
                --log_file logs/full_training_*.log \
                --output_dir analysis/full
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    
    analyze)
        print_header "Analyze Latest Results"
        
        # Find latest log file
        latest_log=$(ls -t logs/*.log 2>/dev/null | head -1)
        
        if [ -z "$latest_log" ]; then
            echo -e "${RED}✗${NC} No log files found in logs/"
            exit 1
        fi
        
        echo -e "${GREEN}▶${NC} Analyzing: $latest_log"
        python scripts/analyze_results.py \
            --log_file "$latest_log" \
            --output_dir analysis/latest
        ;;
    
    tests)
        print_header "Run Test Suite (221 tests, ~23 seconds)"
        pytest -v --tb=short
        ;;
    
    gpu-check)
        print_header "Check GPU Availability"
        python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
else:
    print('No GPU available - using CPU')
"
        ;;
    
    clean)
        print_header "Clean Generated Files"
        echo -e "${YELLOW}⚠${NC}  This will remove checkpoints, logs, and analysis results"
        echo ""
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf checkpoints_*/
            rm -rf logs/*.log
            rm -rf analysis/*/
            echo -e "${GREEN}✓${NC} Cleaned generated files"
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    
    help|*)
        echo ""
        echo "OKADFA Experiment Runner"
        echo ""
        echo "Usage: $0 [experiment]"
        echo ""
        echo "Available experiments:"
        echo "  quick       - Quick test (100 steps, ~3 minutes)"
        echo "  extended    - Extended training (1000 steps, ~30 minutes)"
        echo "  benchmark   - Benchmark comparison (~20 minutes)"
        echo "  full        - Full training (10000 steps, ~5 hours)"
        echo "  analyze     - Analyze latest results"
        echo ""
        echo "Utility commands:"
        echo "  tests       - Run full test suite (221 tests)"
        echo "  gpu-check   - Check GPU availability"
        echo "  clean       - Remove generated files"
        echo "  help        - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 quick              # Run quick test"
        echo "  $0 extended           # Run extended training"
        echo "  $0 analyze            # Analyze latest results"
        echo ""
        exit 0
        ;;
esac

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Experiment Complete! ✓${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
