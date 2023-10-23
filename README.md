# ASSERT
## ASSERT: Automated Safety Scenario Red Teaming for Evaluating the Robustness of Large Language Models
Paper Link: https://arxiv.org/abs/2310.09624
Authors: Alex Mei*, Sharon Levy*, William Yang Wang

<img width="714" src="https://github.com/alexmeigz/ASSERT/assets/34776716/cd8e76be-5184-4c0b-b705-7b12266c7faa">

## Setup
- This project is based on `python v3.11.4` and `pip v23.1.2`
- Install venv: https://realpython.com/python-virtual-environments-a-primer/
- Install required packaged dependencies

## Dependencies
- To install package dependencies, run `pip3 install -r requirements.txt`
- To update package dependencies, run `pip3 freeze > requirements.txt`

## Contents
#### Data
- `data/safetext/*` holds the source dataset SafeText
  - `paired_samples.json` contains the original SafeText dataset
  - `safe_samples.json` and `unsafe_samples.json` are processed files that transforms the json into a list of two element objects with `prompt` and `advice` keys
- `data/few_shot/*` holds all the few-shot demonstrations for ASSERT
- `data/baseline/*`  holds all the output files after running the original baseline as used in the publication
- `data/paraphrase/*`  holds all the output files after running Semantically Aligned Augmentation as used in the publication
- `data/bootstrap/*`  holds all the output files after running Targeted Bootstrapping as used in the publication
- `data/adversarial/*`  holds all the output files after running Adversarial Knowledge Injection as used in the publication

#### Modules
- `models/eval.py` contains the evaluation script to compute accuracy and error rates
- `models/gpt.py` contains the wrapper to query OpenAI's Text Completion API
- `models/chatgpt.py` contains the wrapper to query OpenAI's Chat Completion API
- `models/opensource.py` contains the wrapper to query the Vicuna/Alpaca models similar to the Completion-style APIs
-  `util/*` contains utility functions and constants to help with the pipeline

#### Main 
- `baseline.py` is the file to run the baseline pipeline
- `semantic.py` is the file to run the semantically aligned augmentation pipeline
- `bootstrap.py` is the file to run the targeted bootstrapping pipeline
- `adversarial.py` is the file to run the adversarial knowledge injection pipeline

## Environment Variables
Add a `.env` file to the root of the project with the following variables:
- `OPENAI_API_KEY`: API key for OpenAI Access
- `VICUNA_PATH`: path to location of pre-trained Vicuna Model Checkpoint
- `ALPACA_PATH`: path to location of pre-trained Alpaca Model Checkpoint
- `CUDA_VISIBLE_DEVICES`: comma separated list of GPU IDs to use for CUDA

## Usage
- Create a new venv with `python3 -m venv .venv`
- Activate venv with `source .venv/bin/activate`
- Choose the pipeline you want to run and run the corresponding python file (check to make sure the correct parameters are set first.)

## Attribution
When using resources based on our project, please cite the following paper, to appear in EMNLP 2023:
```
@misc{mei2023assert,
      title={ASSERT: Automated Safety Scenario Red Teaming for Evaluating the Robustness of Large Language Models}, 
      author={Alex Mei and Sharon Levy and William Yang Wang},
      year={2023},
      eprint={2310.09624},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
