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
   - Align with Human Feedback: Reinforcement Learning (RLHF)
   - Evaluate
4. **Application Integration**
   - Optimize and Deploy model for inference
   - Augment model and build LLM-powered applications

## Pre-training

- Model architecture affects how model is trained and what it could do
- Encoder-only Models (Auto-encoding Models)
  - Trained with "Masked Language Modelling"
    - Token (A word) of input sequence (sentence) is masked
    - Model is asked to predict the masked token to reconstruct the original sentence
  - Encoded "Bi-directional context" of the sequence (before and after the token)
    - Understand the full-context of the token, not just the word before the token
  - Best for tasks requiring full context
    - Sentiment Analysis
    - Named Entity Recognition
    - Word Classification
- Encode Decode Models
  - Trained with "Span Corruption" (but different for others)
    - Mask random sequences (more than one tokens) in a sequence
  - Best for tasks
    - Translation
    - Text Summarisation
    - Question Answering
  - E.g. T5, BART
- Decoder-only Models (Auto-regressive Models)
  - Trained with "Causal Language Modelling"
    - Model is asked to predict next token based on previous sequence of tokens
    - Model can only "see" tokens leading up to the token being predicted ("uni-directional")
  - Bet for tasks
    - Text Generation
    - Larger model seem to be "general"
  - E.g. GPT, BLOOM
- Quantisation
  - To reduce the memory required to store and train models
  - By reducing the precision off the model weights
- Chinchilla Scaling Laws
  - Larger doesn't always mean better, maybe it is under-trained
    - Consider more data but fewer parameters (e.g. LLaMA)

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

## Instruction Fine-Tuning

- Supervised Learning
- Train model on example that demonstrate "How it should respond"
  - E.g. "Classify this review: ... ||| positive"
- Fine-tune for a single task
  - Around 500~1000 examples should be enough
  - Catastrophic Forgetting
    - Full fine-tuning modifies the weights of base LLM
    - Improves on the single task, but degrades on others
- Fine-tune for multiple tasks
  - Requires more data (compare to single task)
  - FLAN: Fine-tuned LAnguage Net

## Parameter Efficient Fine-Tuning

> https://www.coursera.org/learn/generative-ai-with-llms/lecture/A6TDx/lab-2-walkthrough

- Update a small subset of parameters OR add a small number of new parameters (layers)
- Selective Method
  - Fine-tune on subset of initial LLM parameters
- Re-parameterization Method
  - Fine-tune by re-parameterising model weights using a low-rank representation
  - LoRA: Low Rank Adaptation
    - https://www.coursera.org/learn/generative-ai-with-llms/lecture/NZOVw/peft-techniques-1-lora 
- Additive Method
  - Add trainable layers or parameters to base LLM model
  - Adapters: Add new trainable layers
  - Soft Prompt: Manipulate inputs to achieve better performance (with frozen parameters)
    - E.g. Prompt tuning
    - https://www.coursera.org/learn/generative-ai-with-llms/lecture/8dnaU/peft-techniques-2-soft-prompts

## Reinforcement Learning from Human Feedback (RLHF)

- Personalised LLM through Feedback?
- Reinforcement Learning:
  - Agent & Environment
  - Objective, Action Space
- ...

## LLM Evaluation

- Similarity?
- Below are N-grams based Evaluations
- ROUGE: Recall-oriented under Study for Jesting Evaluation
  - Text Summarisation
  - Compare generated summary with one or more reference summaries
- BLEU: Bilingual Evaluation Under Study
  - Text Translation
  - Compare generated translation with human translations

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
