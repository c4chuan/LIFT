# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LIFT is a Learning from Interaction Fine-Tuning system for Vision-Language Models (VLMs) designed for web navigation tasks. The project integrates Visual WebArena (VWA) environments, reward-based learning, and VLLM-based model serving to create a complete training pipeline for web agents.

## Architecture

### Core Components

**Environment Management** (`src/environment_manager.py`):
- `EnvironmentManager`: Central coordinator managing multiple VWA browser environments
- `VWATask`: Data structure representing individual web navigation tasks
- Handles parallel environment execution and message queuing

**Model Backend** (`src/vllm_backend.py`):
- `ResponseSampler`: VLLM-based sampling service for generating model responses
- Serves models via FastAPI with configurable sampling parameters
- Supports multi-GPU tensor parallelism

**Reward System** (`src/reward/rewarder.py`):
- `Rewarder`: Vision-language model for evaluating agent actions
- Computes shift_reward and zoom_reward based on action quality
- Supports batch processing and visualization

**Prompt Construction** (`src/prompts/prompt_construct.py`):
- `PromptConstructor`: Builds structured prompts for VLM input
- Integrates task intent, browser screenshots, and action history
- Templates and examples defined in `src/prompts/prompts.py`

### Data Flow

1. **Task Loading**: Web navigation tasks loaded from `data/LIFT_sft.json`
2. **Environment Setup**: Browser environments initialized with VWA configs
3. **Prompt Generation**: Screenshots and task context assembled into VLM prompts
4. **Action Sampling**: Multiple candidate actions generated via VLLM backend
5. **Reward Computation**: Actions evaluated using vision-based reward model
6. **Environment Stepping**: Best action executed in browser environment

## Common Development Tasks

### Running the VLLM Server

Start the model serving backend:

```bash
cd src/shells
bash serve.sh
```

This launches VLLM server on port 7171 with:
- Multi-GPU setup (GPUs 0,1)
- Model: `/data/wangzhenchuan/Projects/LIFT/merged_models/0728_qwen25_vl_7b_instruct_300steps`
- Tensor parallel size: 2
- API key authentication enabled

### Environment Setup

The project uses multiple conda environments:
- `easyr1`: Main environment for VLLM serving and model inference
- Requirements managed in `src/requirements.txt` and `vwa_requirements.txt`

### Running Training/Evaluation

Main training script: `src/main.py`
- Executes single task evaluation with reward-based action selection
- Configurable parameters: `cache_dir`, `results_dir`, number of samples

Environment manager: `src/environment_manager.py`
- Supports parallel environment execution
- Manages task queues and message passing

## Key Configuration

**Model Paths**:
- Base model: `/data/wangzhenchuan/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct`
- Fine-tuned models: `merged_models/` directory
- Reward model uses same base Qwen2.5-VL architecture

**Environment Settings**:
- Browser viewport: 1280x2048 pixels
- Headless mode enabled
- SOM (Set of Marks) action tagging
- Auto-login handling for authenticated sites

**Reward Configuration**:
- Dual reward structure: shift_reward + zoom_reward
- Visual attention-based evaluation
- Configurable visualization output

## Data Structure

**Tasks**: JSON format with fields:
- `task_id`: Unique identifier
- `intent`: Natural language goal description
- `image`: Optional reference images
- `storage_state`: Authentication cookies for websites

**Results**: Organized by date in `outputs/` directory
- Screenshots for each environment step
- Response texts and reward scores
- Debug information and trajectories

## Integration Points

**VWA Integration**: 
- Uses `vwa.src.envs.browser.FastCachedwActionMatchingBrowserEnv`
- Auto-login system for authenticated web environments
- Action caching and replay capabilities

**VLLM Integration**:
- HTTP API on localhost:7451 for sampling
- Qwen-VL model family support
- Configurable generation parameters

**Ray Integration** (`src/ray_reward_backend.py`):
- Distributed reward computation
- Parallel processing of action candidates

## Testing and Validation

**Data Construction**:
- `src/sft_data_construct.py`: SFT dataset preparation
- `src/valid_data_construct.py`: Validation set creation

**Debugging**:
- Extensive logging in environment manager
- Visual debugging in reward system
- Trajectory saving for analysis