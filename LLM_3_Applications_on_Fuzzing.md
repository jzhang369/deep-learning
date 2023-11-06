# Applying LLMs to Software Fuzzing

## Background

Traditional Fuzzers can be classified as

+ mutation-based: iteratively perform transformations on seeds to generate new fuzzing inputs. 
+ generation-based: create complete code snippets using pre-defined grammars and built-in knowledge of the semantics of the target langauge. 

Learing-Based Fuzzing

+ Pre-LLM: to train a neural network to generate fuzzing inputs. 
    + TreeFuzz
    + to fuzz PDF parsers, openCL, C, network protocols, and Javascripts. 

+ Post-LLM: 
    + TitanFuzz uses Codex to generate seed programs and InCoder to perform template-based mutation. 
    + FuzzGPT: leverages historical bug-triggering code snippets to either prompt or directly fine-tune LLMs towards generating more unusual code snippets for more effective fuzzing. 
    + LLM4All: TBA


## 









[video](https://www.youtube.com/watch?v=k9gt7MNXPDY)

[blog article](https://infiniteforest.org/LLMs+to+Write+Fuzzers)

[blog article](https://research.nccgroup.com/2023/02/09/security-code-review-with-chatgpt/)

[blog article](https://security.googleblog.com/2023/08/ai-powered-fuzzing-breaking-bug-hunting.html)

[blog article](https://www.csoonline.com/article/652029/code-intelligence-unveils-new-llm-powered-software-security-testing-solution.html)