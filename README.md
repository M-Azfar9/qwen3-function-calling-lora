# Qwen3 Function-Calling Fine-Tuning with LoRA

An LLM fine-tuning project focused on structured function calling, tool selection, and argument generation.

## Project Overview

Fine-tuned `Qwen/Qwen3-0.6B` using FP16 LoRA and supervised fine-tuning (SFT) on the Glaive Function Calling v2 dataset.

The goal is to improve the model's ability to select the appropriate tool and generate structured JSON arguments from user requests.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers and Datasets
- PEFT (LoRA)
- TRL (SFTTrainer)
- Kaggle GPU notebooks

## Evaluation Results

The base model and fine-tuned model were evaluated on 1,000 test examples.

| Metric | Before Fine-Tuning | After Fine-Tuning | Improvement |
|---|---:|---:|---:|
| Valid JSON Rate | 18.5% | 100.0% | +81.5 percentage points |
| Tool-Name Accuracy | 15.7% | 100.0% | +84.3 percentage points |
| Exact Argument Accuracy | 14.7% | 96.4% | +81.7 percentage points |

### Results Summary

Before fine-tuning, the base model frequently generated natural-language responses instead of structured tool-call JSON. After fine-tuning, the model consistently generated valid JSON and selected the expected tool across the evaluated test examples, with 96.4% exact argument accuracy.

These results demonstrate a substantial improvement on the evaluated benchmark. They do not guarantee equivalent performance on arbitrary real-world prompts or unseen tool schemas.

### Pre-Training Examples

**Example 1**

- Expected tool: `calculate_tip`
- Predicted tool: None
- Output: The model explained the tip calculation in natural language instead of generating a structured tool call.

**Example 2**

- Expected tool: `analyze_website`
- Predicted tool: None
- Output: The model acknowledged the website analysis request but did not produce a structured tool call.

### Post-Training Examples

The fine-tuned model generated structured tool calls such as:

```json
{
  "name": "calculate_tip",
  "arguments": {
    "bill_amount": 50,
    "tip_percentage": 20
  }
}
```

This output contains the tool name and arguments in a machine-readable format.

## Repository Contents

- `lora-tool-selection-project-1.ipynb`: Fine-tuning notebook
- `pre_training_evaluation_results.md`: Base-model evaluation report
- `qwen3-function-calling-lora-final/`: LoRA adapter, tokenizer files, and evaluation artifacts

## Base Model and Dataset

- Base model: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- Dataset: [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

## Loading the Adapter

The adapter requires the original base model. Load the base model and apply the saved adapter using Hugging Face Transformers and PEFT.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen3-0.6B"
adapter_path = "./qwen3-function-calling-lora-final"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

model.eval()
```

## Limitations

- Evaluation results depend on the selected test set and evaluation method.
- Further testing on unseen prompts, different tool schemas, malformed inputs, and requests that do not require tools is needed to assess generalization.
- Exact argument accuracy requires the generated arguments to match the reference arguments exactly.
- The current benchmark evaluates structured tool-call generation; it does not establish that the model can execute tools or reliably decide when a tool should not be called.

## License

Check the base model and dataset licenses before redistributing model weights or dataset-derived artifacts.
