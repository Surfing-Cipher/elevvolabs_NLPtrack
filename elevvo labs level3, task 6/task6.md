# Task 6: Fine-Tuning a Question-Answering Model

## 1. Overview

This task focused on fine-tuning a pre-trained **DistilBERT** model for **extractive question answering**. Using a subset of the **Stanford Question Answering Dataset (SQuAD)**, the goal was to train the model to correctly predict answer spans from a given text passage.

The process included:

- Dataset loading and preprocessing
- Model fine-tuning using the **Hugging Face Trainer API**
- Evaluation on a validation set
- Inference on a completely unseen example

---

## 2. Dataset and Setup

We used the `squad` dataset from Hugging Face Datasets. To ensure faster training and demonstration, we worked with small subsets:

- **Training set:** `train[:5000]`
- **Validation set:** `validation[:500]`

The pre-trained model and tokenizer were both **distilbert-base-uncased** for a balance of performance and efficiency.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load dataset subsets
raw_datasets = load_dataset("squad", split="train[:5000]")
raw_datasets_eval = load_dataset("squad", split="validation[:500]")

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
```

---

## 3. Data Preprocessing

Each question-context pair was tokenized, and the **start** and **end token positions** of the answers were computed.
Key steps included:

- **Truncation & Document Striding:** For contexts longer than `max_length`, overlapping chunks were created with a stride of `128`.
- **Offset Mapping:** Used to map token positions back to character positions in the original text.
- **Answer Labeling:** If the answer span did not appear in the current chunk, the positions were set to `0`.

This ensured the model could learn to predict exact answer spans during training.

---

## 4. Training Configuration

Training was performed using the **Trainer** API for simplicity and automation.
Key hyperparameters:

- **Output Directory:** `./qa_finetuning_output`
- **Learning Rate:** `2e-5`
- **Batch Size:** `16`
- **Epochs:** `3`
- **Weight Decay:** `0.01`
- **Evaluation:** Performed at the end of each epoch

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./qa_finetuning_output",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
```

---

## 5. Training Results

Training was logged with **Weights & Biases (W\&B)**.
Loss values showed consistent improvement across epochs:

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |        No log |          2.2136 |
|     2 |        2.7449 |          1.8820 |
|     3 |        2.7449 |          1.8483 |

The final **training loss** was `2.1235`.
Both the fine-tuned model and tokenizer were saved for later inference.

---

## 6. Inference Test

To evaluate performance, the fine-tuned model was tested on an unseen example.

**Context:**

> The quick brown fox jumps over the lazy dog. The fox is known for its speed and agility, and the dog is very sleepy.

**Question:**

> What is the fox known for?

**Predicted Answer:**

> **speed and agility**

**Details:**

- **Score:** `0.1345`
- **Start Index:** `70`
- **End Index:** `87`

The result shows the model correctly identified the relevant text span from the context, demonstrating successful fine-tuning.

---

## 7. Output Artifacts

The following files were produced:

- `qa_finetuning_output/` — Contains the fine-tuned model and tokenizer.
- `wandb/` — Training logs and metrics tracked through Weights & Biases.
