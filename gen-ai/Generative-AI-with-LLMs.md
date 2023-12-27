# Generative AI with LLMs
<!-- markdownlint-disable MD034 -->

> https://www.coursera.org/learn/generative-ai-with-llms

## GenAI Project Lifecycle

TODO: include the graph

1. **Scope**
   - Define the Use Case
   - As narrowly and accurately as possible
   - E.g. Essay Writing, Text summarisation, Information Retrieval, Invoke API...etc
2. **Select**
   - Choose an existing model
   - Pre-train your onw model
3. **Adapt and Align Model** (iterative)
   - Prompt Engineering
   - Fine-tuning: Supervised Learning
   - Align with Human Feedback: Reinforcement Learning
   - Evaluate
4. **Application Integration**
   - Optimize and Deploy model for inference
   - Augment model and build LLM-powered applications

## Pre-training

...

## Prompt Engineering

- In-Context Learning
  - Setting the context in prompts to make N-shot Inferences
- Zero-shot Inference
  - E.g. in prompt, asking LLM to classify a text
- One-shot Inference
  - E.g. in prompt, give one example of text classification,
    - then asking LLM to classify another unseen one
- Few-shot Inference
  - E.g. in prompt, give two or more examples to text classification,
    - then asking LLM to classify another unseen one
  - If still incorrect inferences after five or six shots, it may not work at all
    - Consider other training techniques

## LLM Configuration

- Top-K Sampling
  - Sample from top K results after applying random-weighted strategy using the probabilities
  - E.g. To predict next token, only consider the top K tokens of highest probabilities
  - Enable some randomness while preventing the selection of highly improbable completion tokens
- Top-P Sampling
  - Sample from top-ranked consecutive results by probability and with a cumulative probability <= p
  - cf with top-k
    - K is on the number of results (tokens) to be chosen from
    - P is the total probability (of tokens) to be chosen from
- Temperature
  - Affect shape of probability distribution
    - Higher means more randomness (more close to uniform distribution among tokens)
    - Lower means less randomness in the output (probability distribution is more central to one token)
