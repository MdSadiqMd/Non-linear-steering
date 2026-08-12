set dotenv-load := true

# Default: show available commands
default:
    @just --list

# Install all dependencies
install:
    uv sync

# Install with dev dependencies
install-dev:
    uv sync --group dev

# Download a model from HuggingFace (default: gpt2)
download-model model="gpt2":
    uv run python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('{{model}}'); AutoTokenizer.from_pretrained('{{model}}'); print('Downloaded: {{model}}')"

# Run all tests
test *args:
    uv run pytest {{args}}

# Run tests with verbose output
test-v:
    uv run pytest -v

# Run a specific test file
test-file file:
    uv run pytest {{file}} -v

# Format code with ruff
fmt:
    uv run ruff format src tests

# Check formatting without changing files
fmt-check:
    uv run ruff format --check src tests

# Lint code with ruff
lint:
    uv run ruff check src tests

# Lint and auto-fix
lint-fix:
    uv run ruff check --fix src tests

# Type check with pyright (if installed)
typecheck:
    uv run pyright src

# Run all quality checks
check: fmt-check lint test

# Check if a model is ready
preflight model="gpt2":
    uv run nls preflight --model {{model}}

# Train a probe on labeled data
train-probe model="gpt2" dataset="data/probe_train.jsonl" layer="6" epochs="3" output="checkpoints/probe.pt":
    uv run nls train-probe \
        --model {{model}} \
        --dataset {{dataset}} \
        --probe-layer {{layer}} \
        --epochs {{epochs}} \
        --output {{output}}

# Train steering
train-steering model="gpt2" prompts="data/prompts.jsonl" probe="checkpoints/probe.pt" inject="4" probe_layer="6" steps="100" output="checkpoints/steering.pt":
    uv run nls train-steering \
        --model {{model}} \
        --prompts {{prompts}} \
        --probe {{probe}} \
        --inject-layer {{inject}} \
        --probe-layer {{probe_layer}} \
        --device cpu \
        --steps {{steps}} \
        --output {{output}}

# Full training pipeline with GPT-2
train-all: (train-probe "gpt2" "data/probe_train.jsonl" "6" "3" "checkpoints/probe.pt") (train-steering "gpt2" "data/prompts.jsonl" "checkpoints/probe.pt" "4" "6" "50" "checkpoints/steering.pt")

# Show CLI help
help:
    uv run nls --help

# Show train-probe help
help-probe:
    uv run nls train-probe --help

# Show train-steering help
help-steering:
    uv run nls train-steering --help

# Inspect a probe checkpoint
inspect-probe path="checkpoints/probe.pt":
    uv run python -c "import torch; c=torch.load('{{path}}', map_location='cpu'); print('Keys:', list(c.keys())); print('Config:', c.get('config')); print('Probe layer:', c.get('probe_layer'))"

# Inspect a steering checkpoint
inspect-steering path="checkpoints/steering.pt":
    uv run python -c "import torch; c=torch.load('{{path}}', map_location='cpu'); print('Keys:', list(c.keys())); [print(f'{k}: {v}') for k,v in c.items() if k != 'state_dict']"

# Clean generated files
clean:
    rm -rf checkpoints/*.pt
    rm -rf .pytest_cache
    rm -rf __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run a quick smoke test (fast subset of tests)
smoke:
    uv run pytest tests/test_model_io.py tests/test_probe.py -v

# Watch tests (requires pytest-watch)
watch:
    uv run ptw -- -v

# Open Python REPL with project loaded
repl:
    uv run python -c "from non_linear_steering import *; print('Loaded: CausalProbe, LayerwiseSteering, etc.'); import code; code.interact(local=locals())"
