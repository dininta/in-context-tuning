import json
import math
import numpy as np
import os
import pandas as pd
import random
import torch
from unidecode import unidecode


N_LABEL = {
    "emo": 4,
    "tweet_eval-hate": 2,
    "climate_fever": 4,
    "health_fact": 4,
    "ethos-gender": 2,
    "ethos-religion": 2,
    "anli": 3,
    "scitail": 2,
    "medical_questions_pairs": 2,
    "paws": 2,
    "tweet_eval-emotion": 4,
    "sick": 3,
    "amazon_polarity": 2,
    "ag_news": 4,
}

def n_label(task_name: str) -> int:
    return None if task_name not in N_LABEL else N_LABEL[task_name]

def random_seed(value):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def get_tasks_list(filename, split_name):
    with open(filename, 'r') as fin:
        split_dict = json.load(fin)
    return split_dict[split_name]

def get_task_prefixes(data_path: str, task_name: str) -> list:
    """Returns all task prefixes (e.g., adversarialqa_32_13) of a task."""
    files = sorted(os.listdir(os.path.join(data_path, task_name)))
    prefixes = []
    for filename in files:
        if not filename.endswith(".tsv"):
            continue
        prefix = "_".join(filename.split("_")[:-1])
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes

def read_dev_data(data_path: str, task_names: list) -> list:
    result = []
    for task_name in task_names:
        prefixes = get_task_prefixes(data_path, task_name)
        for prefix in prefixes:
            train_examples = []
            with open(os.path.join(data_path, task_name, prefix + "_train.tsv"), encoding="utf-8") as fin:
                lines = fin.readlines()
            for line in lines:
                d = unidecode(line).strip().split("\t")
                train_examples.append([d[0], d[1:]])

            dev_examples = []
            with open(os.path.join(data_path, task_name, prefix + "_dev.tsv"), encoding="utf-8") as fin:
                lines = fin.readlines()         
            for line in lines:
                d = unidecode(line).strip().split("\t")
                dev_examples.append([d[0], d[1:]])

            result.append({
                "task_name": task_name,
                "task_prefix": prefix,
                "train_examples": train_examples,
                "dev_examples": dev_examples,
            })
    
    return result

def read_test_data(data_path: str, task_names: list) -> list:
    result = []
    for task_name in task_names:
        prefix = get_task_prefixes(data_path, task_name)[0]  # only use 1 prefix for test set

        # Combine train and dev set into dev only
        dev_examples = []
        with open(os.path.join(data_path, task_name, prefix + "_train.tsv"), encoding="utf-8") as fin:
            lines = fin.readlines()
        with open(os.path.join(data_path, task_name, prefix + "_dev.tsv"), encoding="utf-8") as fin:
            lines.extend(fin.readlines())
        for line in lines:
            d = unidecode(line).strip().split("\t")
            dev_examples.append([d[0], d[1:]])

        test_examples = []
        with open(os.path.join(data_path, task_name, prefix + "_test.tsv"), encoding="utf-8") as fin:
            lines = fin.readlines()
        for line in lines:
            d = unidecode(line).strip().split("\t")
            test_examples.append([d[0], d[1:]])

        result.append({
            "task_name": task_name,
            "dev_examples": dev_examples,
            "test_examples": test_examples,
        })
    
    return result

def sample_demos(examples: list, k: int, n_label: int, ex_per_label: int = 16) -> list:
    if n_label is None:
        return random.sample(examples, k)

    # If classification, need to sample from all labels
    result = []
    st = 0
    for i in range(n_label):
        ed = st + ex_per_label
        k_i = math.floor(k / n_label) + (1 if i < (k % n_label) else 0)
        result.extend(random.sample(examples[st:ed], k_i))
        st = ed
    return result

def create_input_text(demos: list, input_text: str, io_sep: str, ex_sep: str) -> str:
    demonstrations = ex_sep.join(['{} {} {}'.format(ex[0], io_sep, random.choice(ex[1])) for ex in demos])
    if input_text is None:
        return demonstrations
    return '{}{}{} {}'.format(demonstrations, ex_sep, input_text, io_sep)

def read_prompt_prefix_dict(filename: str) -> dict:
    df = pd.read_csv(filename, sep='\t', header=0)
    return pd.Series(df.prompt.values,index=df.task_prefix).to_dict()

def read_instruction_dict(filename: str) -> dict:
    instructions_dict = {}
    with open(filename) as fin:
        lines = fin.readlines()
    for line in lines:
        splits = line.strip().split('\t')
        instructions_dict[splits[0]] = splits[1], splits[2]
    return instructions_dict
