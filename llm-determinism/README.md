# llm-determinism

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Experiments on LLM-determinism, with the following goal:

__To run basic experiments using local and small LLMs, capturing and analysing token probabilities, and the variance in output, subject to different hyperparameters. All of this in python.__

## Hypothesis

* Models can be made near deterministic, same input -> same output
* For common topics/prompts, responses are more deterministic subject to small changes in input
* For rare topics/prompts, responses are more stochastic subject to small changes in input.
* Ensemble voting models can be used for structured output to improve determinism.

## Technical assumptions (can be tested)

* Phi (or whatever) via huggingface transformers and ollama produces identical outputs.

## Plan

* Use a simple notebook first to test this, may want to use a different model. 
* Use small, as this will require quite a few iterations.
* Set up repo
* Set up dependencies and notebooks (what about google colab or something?)
* See what else can be used?

## Candidate LLM models

### Qwen2
*   **Parameters**: 0.5B, 1B, 7B versions
*   Scalable and suitable for various tasks
*   0.5B version ideal for super lightweight applications

### Gemma2
*   **Parameters**: 2B, 9B, and 27B versions
*   **Context length**: 8,192 tokens
*   Designed for local deployment and real-time applications

### LaMini-GPT
*   **Parameters**: 774M - 1.5B
*   Optimized for multilingual and instruction-following tasks
*   Efficient in resource-constrained environments

### MiniCPM
*   **Parameters**: 1B - 4B
*   Balanced performance, especially for English and Chinese language processing
*   Competes with larger models like Mistral-7B and LLaMA 7B

### TinyLlama
*   **Parameters**: 1.1B
*   Efficient for mobile and edge devices
*   Performs well in commonsense reasoning tasks

### Phi-3.5
*   **Parameters**: 3.8B
*   Long context length (128K tokens)
*   Multilingual capabilities

### Mistral 7B
*   **Parameters**: 7B
*   Known for its balance of power and efficiency
*   Suitable for both experimental and production environments

### Llama 3.1 8B
*   **Parameters**: 8B
*   Balanced power and efficiency
*   Open source with usage restrictions

## Getting started

See `notebooks/00_20250219_initial_tests.ipynb`.


## Appendix

### Perplexity prompts and results

#### Local LLM logprob prompt

Note that the example given does not work. Refer to  `notebooks/00_20250219_initial_tests.ipynb` for a working example.

> Can you show examples with phi2 of retrieving log probabilities while also changing all the possible parameters that affect determinism]

Certainly! Here's an example of how to retrieve log probabilities with Phi-2 while adjusting parameters that affect determinism:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Input text
input_text = "The capital of France is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Set parameters affecting determinism
gen_kwargs = {
    "max_new_tokens": 20,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 2,
    "return_dict_in_generate": True,
    "output_scores": True,
    "num_return_sequences": 1,
    "pad_token_id": tokenizer.eos_token_id
}

# Set a fixed seed for reproducibility
torch.manual_seed(42)

# Generate text and get log probabilities
with torch.no_grad():
    outputs = model.generate(input_ids, **gen_kwargs)

# Extract generated tokens and scores
generated_tokens = outputs.sequences[0, input_ids.shape[-1]:]
scores = outputs.scores

# Calculate log probabilities
log_probs = torch.stack([torch.log_softmax(score, dim=-1) for score in scores])
token_log_probs = torch.gather(log_probs, -1, generated_tokens.unsqueeze(-1)).squeeze(-1)

# Print results
print("Generated text:")
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
print("\nTokens and their log probabilities:")
for token, log_prob in zip(generated_tokens, token_log_probs):
    print(f"Token: {tokenizer.decode(token)}, Log Probability: {log_prob.item():.4f}")
```

This example demonstrates how to:

1. Set a fixed seed for reproducibility
2. Adjust temperature (controls randomness)
3. Use top-p (nucleus) sampling
4. Apply top-k sampling
5. Set a repetition penalty
6. Use no_repeat_ngram_size to prevent repetition of n-grams
7. Retrieve log probabilities for each generated token

By modifying these parameters, you can experiment with different levels of determinism in the model's output while still accessing the log probabilities[1][4][6].

To make the model even more deterministic, you could:

1. Set `do_sample=False` to use greedy decoding
2. Lower the temperature (e.g., 0.1)
3. Decrease top_p and top_k values
4. Increase the repetition penalty

Remember that while these strategies increase determinism, they may also reduce the model's creativity and flexibility in generating responses[5].

Citations:
[1] http://www.phidot.org/software/mark/docs/book/pdf/chap4.pdf
[2] https://www.dcs.gla.ac.uk/Keith/Chapter.6/Ch.6.html
[3] https://www.datacamp.com/tutorial/phi-2-model
[4] https://huggingface.co/docs/transformers/en/internal/generation_utils
[5] https://www.promptingguide.ai/models/phi-2
[6] https://huggingface.co/docs/transformers/en/model_doc/phi
[7] https://www.researchgate.net/figure/Conditional-probability-of-solving-the-item-correctly-phi-from-2PL-and-the_fig4_327970858
[8] https://crfm.stanford.edu/fmti/May-2024/company-reports/Microsoft_Phi-2.html

---
Answer from Perplexity: pplx.ai/share