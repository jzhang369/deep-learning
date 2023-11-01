# GPT





Objectives:

+ Large language models are trained (using large datasets) to solve common language problems like text classification, question answering, document summarization, and text generation. 

+ They can be *tailored* to solve specific problems in different fields like retail, finance, and etc. (using a relative small field dataset). 


Features:

+ Large
    + Large training dataset
    + Large number of parameters
+ General Purpose
    + Capable of solving general problems
    + Should be directly used instead of being trained
        + Resource restriction of individual users
+ Pre-Trained and Fine-Tuned
    + General models are trained using large dataset
    + Specific models are tuned based on general models using relative small dataset

Benefits:

+ A single model for different tasks
+ The fine-tune process requires minimal field data such as few/zero shot
+ The performance of the model continuously grows when you add more data and parameters


Model Examples:

+ LaMDA
+ PaLM
+ GPT
+ etc.

LLM Development v.s. Traditional Development:

+ LLM Development using pre-trained APIs
    + no ML expertise needed
    + no training examples needed
    + no need to train a model
    + **focus on prompt design**
+ Traditional ML development
    + yes ML expertise needed
    + yes training examples needed
    + yes need to train a model
        + data
        + hardware
        + time
    + **focus on minimizing a loss function**


Prompt Design and Engineering

+ Prompt Design: Prompts involve instructions and context passed to a language model to achieve a desired task.
+ Prompt Engineering is the practice of developing and optimizing prompts to efficiently use language models for a variety of applications. 


Three Types of LLMs, each needs prompting in a different way

+ Generic (or Raw) Language Models: predict the next word/token given a sequence of words/tokens
    + I am a -> professor
+ Instruction Tuned Language Models: predict a response to the instructions 
    + Summarize a text of X
    + Generate a poem in the style of X
    + Give me a list of keywords based on semantic similarity of X
    + Classify the following text into neutral, negative, or positive
+ Dialog Tuned Language Models: have a dialog by predicting the next response
    + Chatbot


Tuning

+ The process of adapting a model into a new domain or set of custom use cases by training the model on new data. 
    + Example: collect training data and tune the LLM for the legal or medical domain. 
+ Fine Tuning: expensive and not realistic in many cases
+ Parameter-Efficient Tuning Methods (PETM): methods for tuning an LLM on your own custom data without duplicating the model. 
    + The base model itself is not altered. 
    + A small number of add-on layers are tuned, which can be swapped in and out at inference time. 


## ChatGPT Prompt Engineering

Principles of Prompting
+ Write clear and specific instructions.
    + clear $\neq$ short
+ Give the model time to think. 
    + Tactic 1: Specify steps to complete a task
        + Step 1: ...
        + Step 2: ...
        + ...
        + Step N: ...
    + Tactic 2: Instruct the model to work out its own solution before rushing into a conclusion. 



## About Langchain

[Video](https://www.youtube.com/watch?v=aywZrzNaKjs)

Langchain

+ An open source framework that allows AI developers to combine LLMs like GPT-4 with external sources of computation and data. 