#!/bin/bash

# Default values - these can be overridden by environment variables
DEFAULT_MODEL="qwen2_5_vl"
DEFAULT_TASKS="megabench_open"
DEFAULT_BATCH_SIZE="1"
DEFAULT_PORT="12345"
DEFAULT_MAX_PIXELS="376320"
DEFAULT_LOG_SAMPLES="true"
DEFAULT_OUTPUT_PATH="./logs/"
DEFAULT_DRY_RUN="false"

# Read from environment variables if set, otherwise use defaults
MODEL=${LMMS_MODEL:-$DEFAULT_MODEL}
TASKS=${LMMS_TASKS:-$DEFAULT_TASKS}
BATCH_SIZE=${LMMS_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
PORT=${LMMS_PORT:-$DEFAULT_PORT}
MAX_PIXELS=${LMMS_MAX_PIXELS:-$DEFAULT_MAX_PIXELS}
LOG_SAMPLES=${LMMS_LOG_SAMPLES:-$DEFAULT_LOG_SAMPLES}
OUTPUT_PATH=${LMMS_OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}
DRY_RUN=${LMMS_DRY_RUN:-$DEFAULT_DRY_RUN}

export HF_HOME=/home/huanan/.cache/huggingface/

# Override with command-line arguments if provided
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --tasks) TASKS="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --max-pixels) MAX_PIXELS="$2"; shift ;;
        --log-samples) LOG_SAMPLES="true" ;;
        --output-path) OUTPUT_PATH="$2"; shift ;;
        --dry-run) DRY_RUN="true" ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model MODEL           Model to use (default: $DEFAULT_MODEL)"
            echo "  --tasks TASKS           Tasks to run (default: $DEFAULT_TASKS)"
            echo "  --batch-size SIZE       Batch size (default: $DEFAULT_BATCH_SIZE)"
            echo "  --port PORT             Main process port (default: $DEFAULT_PORT)"
            echo "  --max-pixels PIXELS     Max pixels (default: $DEFAULT_MAX_PIXELS)"
            echo "  --log-samples           Enable log samples (default: $DEFAULT_LOG_SAMPLES)"
            echo "  --output-path PATH      Output path for logs (default: $DEFAULT_OUTPUT_PATH)"
            echo "  --dry-run               Show the command without executing it"
            echo "  --help                  Show this help message and exit"
            echo ""
            echo "Environment variables (override defaults):"
            echo "  LMMS_MODEL, LMMS_TASKS, LMMS_BATCH_SIZE, LMMS_PORT,"
            echo "  LMMS_MAX_PIXELS, LMMS_LOG_SAMPLES, LMMS_OUTPUT_PATH, LMMS_DRY_RUN"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Set model args based on the model
case $MODEL in
    "qwen2_5_vl")
        MODEL_ARGS="pretrained=Qwen/Qwen2.5-VL-3B-Instruct,max_pixels=$MAX_PIXELS,interleave_visuals=False"
        ;;
    "qts_plus_3b")
        MODEL_ARGS="pretrained=/home/huanan/work/projects/QTS-3B/model-1112/,max_pixels=$MAX_PIXELS"
        ;;
    *)
        # For custom models, check if model args are provided via env var
        if [[ -n "${LMMS_MODEL_ARGS}" ]]; then
            MODEL_ARGS="${LMMS_MODEL_ARGS}"
        else
            echo "Error: Unknown model '$MODEL' and no LMMS_MODEL_ARGS provided"
            exit 1
        fi
        ;;
esac

# Build the command
CMD="accelerate launch --main_process_port=$PORT -m lmms_eval"
CMD+=" --model $MODEL"
CMD+=" --model_args=\"$MODEL_ARGS\""
CMD+=" --tasks $TASKS"
CMD+=" --batch_size $BATCH_SIZE"

# Add optional parameters if enabled
if [[ "$LOG_SAMPLES" == "true" ]]; then
    SUFFIX="${MODEL}_${TASKS}"
    CMD+=" --log_samples"
    CMD+=" --log_samples_suffix $SUFFIX"
    CMD+=" --output_path $OUTPUT_PATH"
fi

# Display the command
echo "Running: $CMD"
echo ""

# Execute the command
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run mode - command not executed"
else
    eval $CMD
fi
