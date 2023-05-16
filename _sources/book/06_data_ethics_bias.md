# Data Ethics and Bias

## *Hello World* Discussion

- Power
- Data

## Lies, Damned Lies, and Statistics

> "If you can’t prove what you want to prove, demonstrate something else and
> pretend they are the same thing. In the daze that follows the collision of
> statistics with the human mind, hardly anyone will notice the difference."
> -- Darrell Huff, *How to Lie with Statistics*, 1954

Since the dawn of statistics in the 17th century statistics have been used used
to guide and mislead. Here we'll discuss a few of the ways issues can arise
when working with datasets.

- Garbage In, Garbage Out
  - No amount of statistical work can make up for unreliable or missing data.
  - Don't assume data independence.
- Tests are Imperfect
  - False negatives
  - False positives
- Pictures Can Be Deceiving
  - Examples via [Finding Examples of Misleading and Deceptive Graphs](https://www.forbes.com/sites/naomirobbins/2012/03/13/finding-examples-of-misleading-and-deceptive-graphs/?sh=4b68872b73c6)
  - ![What's wrong with this figure?](images/bad_figure_0.jpg)
  - ![What's wrong with this figure?](images/bad_figure_1.jpg)
  - ![What's wrong with this figure?](images/bad_figure_2.jpg)
  - ![What's wrong with this figure?](images/bad_figure_3.png)
- *Cum Hoc Ergo Propter Hoc*
  - Correlation is when two variables move via some relationship
    - Positive correlation when they move in the same direction
    - Negative correlation when they move in opposite directions
    - Zero correlation, there is no relationship
  - [Correlation vs. causation](https://tylervigen.com/spurious-correlations)
- Statistical Measures Don't Tell the Whole Story
  - Look at the raw data
  - Data reduction (be wary of extrapolation)
- Sampling Bias
  - Non-response bias
  - Convenience or accidental sampling
- Context Matters
  - Statistics must be thought of in the wider context

## Large Language Models

Large language models (LLM) are machine-learned models trained on extremely
large datasets through the process of deep learning. Generally, an LLM is
distinguished from a standard language model by its conversational proficiency
and reasoning capabilities.

Notable LLMs include OpenAI's GPT-3, ChatGPT, and GPT-4, Google's Bard, and
Meta's LLaMA. Microsoft's Bing Chat uses GPT-4.

| Company | LLM Link | Notes |
| ------- | -------- | ----- |
| OpenAI  | [ChatGPT](https://chat.openai.com/) | Requires free account. |
| Google  | [Bard](https://bard.google.com) | Requires free account. |
| Microsoft | [Bing Chat](https://bing.com/chat) | Requires free account and the Microsoft Edge browser. |
| Meta | [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) | Requires application to private beta. |

Use of these models has prompted many discussions on how they can be used
constructively, how they can be abused, and how to effectively manage their
biases.

