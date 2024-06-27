# ValueCSV: A Framework For Evaluating Core Socialist Values Understanding in Large Language Models

## Overview

To address the potential social risks and safety challenges associated with Large Language Models (LLMs), human values alignment has been proposed to guarantee LLMs' outputs align with human values
as a key step to achieve responsible AI technology. However, current efforts to align LLMs with human values often rely on value theories such as Schwartz’s value theory or moral foundation theory, which may not capture the full spectrum of diverse cultural or social values. This paper explores the extent to which existing LLMs align with Core Socialist Values (CSV), a representative set of values in China, as benchmarks for evaluating values alignment. Our framework is publicly available at [here](https://github.com/ValueCSV).

<p align="center">
    <img src="https://github.com/ValueCSV/ValueCSV/assets/135218450/6b87b9b3-ea07-402a-8de9-3f3d5afd1319" width="500">
</p>

<br>

In this repository, we provide:

- ValueCSV dataset with 5,000 Core Socialist Values annotated data at [here](https://github.com/ValueCSV/ValueCSV/blob/main/Dataset/ValueCSV5000.xlsx).

- Code for training CSV evaluator. Flie [Multi_evaluator.py](https://github.com/ValueCSV/ValueCSV/blob/main/Code/Multi_evaluator.py) for training Multi-label classifier and file [Binary_evaluator.py](https://github.com/ValueCSV/ValueCSV/blob/main/Code/Binary_evaluator.py) for training seperate binary classifiers.

- The checkpoints of trained ValueCSV evaluator.

- A question dataset with 100 question covering diversity CSV dimensions for testing LLM's CSV understanding ability at [here](https://github.com/ValueCSV/ValueCSV/blob/main/Dataset/quest100.xlsx).

## Definition of CSV
Core Socialist Values contains 12 distinct types of values, which are Prosperity, Democracy, Civility, Harmony, Freedom, Equality, Justice, Rule of Law, Patriotism, Dedication, Integrity and Friendliness. These 12 dimensions of values can be categorised into three higher-level groups, i.e., National level, Society level, and Personal level, as listed below:

<p align="center">
    <img src="https://github.com/ValueCSV/ValueCSV/assets/135218450/0dec3790-1d36-4204-b264-7344a6814886" width="500">
</p>

## Pretrained Model

We use both [bert-baes-chinese](https://huggingface.co/google-bert/bert-base-chinese) and [chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) as backstones of evaluators.  

## Experiment Results

The comparison of different ValueCSV evaluators are shown in table below:

<p align="center">
    <img src="https://github.com/ValueCSV/ValueCSV/assets/135218450/8b02c27f-b1a4-47ed-aaf8-996dda95a4ef" width="500">
</p>

Note that M.Value-BERT and M.Value-RoBERTa are both multi-label classifier and Value-BERT and Value-RoBERTa are both 12 separate binary classifiers.


## Checkpoints 

We also release the checkpoints of ValueCSV evaluator Value-Bert at [Google-Drive](https://drive.google.com/drive/folders/1KlXGw6KXA-YG4qs6Mv73qYXn8rEoxaXT?usp=sharing) as Value-Bert outperforms in all ValueCSV evaluators. 

## Data License

We make the dataset under the following licenses:
*  Attribution 4.0 International (CC BY 4.0) license. 
(License URL: https://creativecommons.org/licenses/by/4.0/](https://github.com/ValueCSV/ValueCSV/blob/main/LICENSE)

