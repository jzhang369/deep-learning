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


## FUZZ4ALL

Part 1: Generate The Best Prompt
+ user input: any document that describes the fuzzing inputs to be generated such as document of the SUT, example code snippets, or specifications. 
+ autoprompting: distills user input into a concise but informative prompt for fuzzing using a large, state-of-the-art distillation LLM to sample multiple different candidate prompts. 
    + use a distillation LLM to reduce the given user inputs. 
+ code snippet generation 1: each candidate prompt is passed on to the generation LLM to generate code snippets (i.e., fuzzing inputs).
    + StarCoder. 
+ prompt selection: select the **best** prompt that produces the highest quality fuzzing inputs. 

Part 2: Generate Fuzzing Inputs
+ code snippet generation 2: using the selected, best prompt to continuously sample teh generation LLM to generate fuzzing inputs. 
+ prompt update: to avoid generating many similar fuzzing inputs, it continuously updates the input prompt in each iteration. Specifically, it selects a previously generated input as an example, which demonstrates the kind of future inputs we want the model to generate. In addition, it also append a generation instruction to the initial prompt. 
+ monitor: check whether the SUT misbehaves such as crashes. 










[video](https://www.youtube.com/watch?v=k9gt7MNXPDY)

[blog article](https://infiniteforest.org/LLMs+to+Write+Fuzzers)

[blog article](https://research.nccgroup.com/2023/02/09/security-code-review-with-chatgpt/)

[blog article](https://security.googleblog.com/2023/08/ai-powered-fuzzing-breaking-bug-hunting.html)

[blog article](https://www.csoonline.com/article/652029/code-intelligence-unveils-new-llm-powered-software-security-testing-solution.html)