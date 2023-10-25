# BERT

BERT is an encoder-only transformer architecture. 
+ It is trained to learn language representations. 
+ It mainly differs from the precedent language models because its learned representations contain context from both sides of the sentences (left and right from the word itself). 


To represent each token by an embedding vector, where known methods include
+ Word2Vec
+ Glove

BERT is a *contextualized word embedding* method, where it embeds a word by investing word sequences before and after this word. See the word of "apple" below. 
+ I want to each an apple. 
+ I want to buy an apple watch. 


On a high level, BERT consists of three modules:

+ Embedding: This module converts an array of one-hot encoded tokens into an array of vectors representing the tokens.
+ A stack of encoders: These encoders are the Transformer encoders. They perform transformations over the array of representation vectors
+ Un-embedding. This module converts the final representation vectors into one-hot encoded tokens again.
    + The un-embedding module is necessary for pretraining, but it is often unnecessary for downstream tasks. Instead, one would take the representation vectors output at the end of the stack of encoders, and use those as a vector representation of the text input, and train a smaller model on top of that. 









## BERT Training

BERT is trained using two objectives:
+ Some tokens from the input sequence are masked and the model learns to predict these words (Masked language model).
+ Two “sentences” are fed as input and the model is trained to predict if one sentence follows the other one or not (next sentence prediction NSP).




