# Generative AI for Beginners (GitHub)

> https://github.com/microsoft/generative-ai-for-beginners

## LLM Possible Applications

- Text Summarisation
  - From a block of text (unstructured data) to insights
- Code Explaining
  - From a chunk of code, explain the structure...etc
- Creative Idea Generation
  - Using randomness nature, generate article, essay or assignment...etc
- Question Answering
  - Ask the agent a question and got response 
  - Search engine for specific information
- Text or Code Completion, i.e. Writing Assistance

TODO:

- Brainstorm for applications of your use case
  - Q: Problem?
  - Q: How I would use LLM?
  - Q: Impact?

## Approaches to Tailor LLM

- Prompt engineering with context
  - In the prompt to LLM, provide enough context to ensure proper response
  - The better frame the query, the more accurate response from LLM. I.e. good enough context
- Retrieval Augmented Generation, RAG.
  - Store data in a vector database, then retrieve it and include in the prompt to LLM
- Fine-tuned model
  - Further train the model on data (of expected response) to ensure proper response

## Prompt Engineering

### Best Practices

- Evaluate the latest model
- Separate instructions and context
  - E.g. using delimiter like "`" to distinguish the context
- Be specific and clear
  - Create your own reusable templates
  - Give details about the desired outcome, e.g. length, style, format ...etc
- Be descriptive with examples
  - I.e. "Show and Tell" approach
  - Try Zero-shot, then "Few-shot"...etc
- Use cues to jumpstart completions
  - At the end of your context, provide a few leading words/phrases to the expected outcome
- Double down to emphasize
  - Give instructions before and after the context, this might emphasize the ideas to LLM
- Order of information matter, i.e. "Recency Effect"
- Give the model an "Out" option
  - I.e. try to limit it response so that it does not fabricate statements
  - E.g. just say "I don't know"
- Consider "step to step" solution

### From Personal Experiences

- When there's a role in real-life, try start with "Act as ..."
- ..
