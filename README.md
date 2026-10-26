# Qwen3 Function-Calling Fine-Tuning with LoRA

An LLM fine-tuning project focused on structured function calling, tool selection, and argument generation.

## Project Overview

Fine-tuned `Qwen/Qwen3-0.6B` using FP16 LoRA and supervised fine-tuning (SFT) on the Glaive Function Calling v2 dataset.

The goal is to improve the model's ability to select an appropriate tool and generate structured JSON arguments.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers and Datasets
- PEFT (LoRA)
- TRL (SFTTrainer)
- Kaggle GPU notebooks

## Reported Evaluation Results

On a 1,000-example test evaluation:

| Metric | Result |
|---|---:|
| Valid JSON rate | 100% |
| Tool-name accuracy | 100% |
| Exact argument accuracy | 96.4% |

These results are from the project's recorded evaluation. They should not be interpreted as a guarantee of performance on arbitrary real-world prompts.

## Repository Contents

- `lora-tool-selection-project-1.ipynb`: Fine-tuning notebook
- `pre_training_evaluation_results.md`: Pre-training evaluation report
- `qwen3-function-calling-lora-final/`: LoRA adapter, tokenizer files, and evaluation artifacts

## Base Model and Dataset

- Base model: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- Dataset: [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

## Loading the Adapter

The adapter requires the original base model. Load the base model and apply the saved adapter using Hugging Face Transformers and PEFT.

## Limitations

The benchmark measures performance on a particular test set. Further evaluation on unseen prompts, different tool schemas, malformed inputs, and requests that do not require tools is needed to assess generalization.

## License

Check the base model and dataset licenses before redistributing model weights or dataset-derived artifacts.
