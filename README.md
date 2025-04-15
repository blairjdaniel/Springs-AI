# AI Assistant Springs Project

## Project Task
The task for this project is to create an AI assistant named Kelsey capable of handling customer interactions for Springs RV Resort. The assistant is designed to generate responses to emails, complete forms, and assist with inquiries in a friendly and professional tone.

## Dataset
The dataset used for this project is the **Bitext Customer Support LLM Chatbot Training Dataset**, sourced from Hugging Face. This dataset is designed for fine-tuning large language models for customer support tasks. It includes:

- **Use Case:** Intent Detection
- **Vertical:** Customer Service
- **Specifications:**
  - 27 intents assigned to 10 categories
  - 26,872 question/answer pairs (approximately 1,000 per intent)
  - 30 entity/slot types
  - 12 different types of language generation tags
- **Applications:** Automotive, Retail Banking, Education, Hospitality, Real Estate, Telecommunications, and more.

The dataset was generated using a hybrid methodology combining natural texts, NLP technology for seed extraction, and NLG technology for text expansion. This ensures high-quality, domain-specific data for fine-tuning.
- **Synthetic Data:** 1,000 rows of synthetic data generated to simulate common customer scenarios and responses.
- **Fine-Tuned Data:** Data specifically curated to train the model in the tone and style of Springs RV Resort's sales assistant, Kelsey.

## Pre-trained Model
The pre-trained model selected for this project is `gpt2` from Hugging Face. GPT-2 is a generative language model that excels at text generation tasks. It was fine-tuned on the curated dataset to align with the specific tone and style required for the assistant.

## Performance Metrics
The performance of the fine-tuned model is evaluated using the following metrics:
- **Perplexity:** Measures how well the model predicts the next word in a sequence. Lower perplexity indicates better performance.
- **BLEU Score:** Evaluates the quality of generated text by comparing it to reference responses.
- **Accuracy:** Measures the percentage of correct responses in classification tasks (e.g., intent recognition).
- **F1 Score:** Balances precision and recall to evaluate the model's performance in generating relevant and accurate responses.

### Evaluation Results
| Metric      | Value  |
|-------------|--------|
| Perplexity  | TBD    |
| BLEU Score  | TBD    |
| Accuracy    | TBD    |
| F1 Score    | TBD    |

### Interpretation
- The model's performance metrics will be updated after evaluation on the test dataset.
- The goal is to achieve high accuracy and F1 scores while maintaining a low perplexity for natural and coherent responses.
