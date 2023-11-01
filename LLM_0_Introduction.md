# Large Language Models

## About LLM

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*RYNNKmmi1ShV7xx76qtXww.png)







## Two Different Models

+ BERT: Representation Embedding Model
+ GPT: Generative AI model


![](https://media.geeksforgeeks.org/wp-content/uploads/20230321032520/bart1drawio-(2).png)


## Usage of LLM Models

There are two different strategies to use these two types of models, respectively.

+ Fine Tuning: By adding heads or [adapters](https://adapterhub.ml/) to BERT. Then the entire model is trained for specific tasks - the BERT model uses pre-trained parameters, but the heads/adapters use random initialized parameters. it is worth noting the heads imply the BERT parameters to be adjusted too, although with only small changes expected. Comparatively, adapaters do not expect the BERT parameters to be changed at all. Therefore, adapters themselves can be trained, removed, and later re-deployed; they are small in size. This fine-tuning strategy also falls into the transfer learning, where you can find more discussion about transfering using BERT [here](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html).

+ Prompt Engineering: The model is generative and the prompt will decide what to be generated. 
    + Instruction Learning: the prompt describes what to be wanted.
    + In-context learning: the prompt offers examples that the next question wants to follow. 
    + Few shot
    + One shot
    + Zero shot
    + ...














Notes
+ encoder, decoder, transformer
