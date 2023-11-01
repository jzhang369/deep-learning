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


## BERT Usage : Fine Tuning

+ input: a sequence of vectors
+ output: a sequence of vectors

The input and the output have the same size.  

BERT is mainly used by *Fine-Tuning* for *downstream* tasks. 

### Application 1

+ Input: sequence
+ Output: class
+ Example: sentiment analysis

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*nB3Ltz0FuRqORe9lWJMfXA.png)

The linear model (i.e., the logistic regression model, model #2) has random initialization; the BERT model (i.e., model #1) is initialized by pre-train. Both the logistic regression model and the BERT model will be trained together. 


### Application 2

+ Input: sequence
+ Output: sequence, where len(input) == len(output)
+ Example: POS Tagging

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*dF1IfTFFlDLwt6VjS8TGow.jpeg)

![](https://classic.d2l.ai/_images/bert-tagging.svg)

### Application 3

+ Input: sequence1, seqence2
+ Output: a class (i.e., contradiction, entailment, or neutral)
+ Example: to detect whether sequence1 is the premise of sequence2

[Video, 33:40](https://www.youtube.com/watch?v=gh0hewYkjgo&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=19)


### Application 4

Extraction-Based Question Answering (QA)

+ Input: Document = {d1, d2, .., dN}, Query = {q1, q2, ..., qM}
+ Output: {s, e} to integers to indicate the starting and ending index of the answer

Two vectors to be learnt, one is for starting and another one for ending, both of which are radomly initialized. 

[Model](https://youtu.be/gh0hewYkjgo?list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&t=2306)


### Application 5

Seq2Seq model: the T5 model

## Progress

[Video, 27:30](https://www.youtube.com/watch?v=gh0hewYkjgo&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=19)