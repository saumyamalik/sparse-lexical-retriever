import os
import time
import sys
import torch
import logging
import json
import numpy as np
import random
import pickle
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import AutoTokenizer, OPTModel

from beir_helper import load_model

def print_topk(weights, k, tokenizer, largest=True):
    top = torch.topk(weights, k, largest=largest)
    top_vals = top[0]
    top_indices = top[1]
    print('Values:')
    print(top_vals)
    print('Tokens:')
    print(tokenizer.decode(top_indices))
    print('Mean:', top_indices.mean(dtype=torch.float32))
    return top_vals, top_indices

def get_topk(weights, k, tokenizer, largest=True):
    top = torch.topk(weights, k, largest=largest)
    top_vals = top[0]
    top_indices = top[1]
    return top_vals, top_indices
    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    best_path = "LR.0.05/checkpoint"
    best_step = "step-325"
    my_model, _, _ = load_model(best_path, best_step)
    model_weights = my_model.weights
    
    worst_path = "LR.0.0001/checkpoint"
    bad_model, _, _ = load_model(worst_path, best_step)
    bad_weights = bad_model.weights
    bm_25weights = torch.load('./bm25/weights')['IDF']
    
    k = 100
    
    _, ordering = get_topk(bm_25weights ,tokenizer.vocab_size, tokenizer)
    
    print('Top Model weights: k=', k)
    top_vals, top_indices = get_topk(model_weights ,k, tokenizer)
    bottom_vals, bottom_indices = get_topk(model_weights, k, tokenizer, largest=False)
    
    ranks = []
    print(bottom_vals)
    for index in bottom_indices:
        hot = ordering.where(ordering==index, 0)
        rank = torch.argwhere(hot)
        ranks.append(rank)
        
    print("ranks:")
    print(ranks)
    print(torch.tensor(ranks).mean(dtype=torch.float32))
    

  
    print('Middle BM25 weights: k=', k)
    top_vals, top_indices = print_topk(model_weights ,k, tokenizer)
    middle_indices = top_indices[-100:]
    # print(tokenizer.decode(top_indices[-100:]))
    
    # # print('---------------')
    
    # print('Bottom BM25 weights: k=', k)
    # print_topk(bm_25weights ,k, tokenizer, largest=False)
    
    # print('Bottom model weights: k=', k)
    # print_topk(model_weights ,k, tokenizer, largest=False)

    model_abs_weights = torch.abs(model_weights)
    # print('Bottom model weight magnitude: k=', k)
    _, bottom_abs_indices = get_topk(model_abs_weights ,k, tokenizer, largest=False)
    for index in middle_indices:
        hot = ordering.where(ordering==index, 0)
        rank = torch.argwhere(hot)
        ranks.append(rank)
        
    print("ranks:")
    print(ranks)
    print(torch.tensor(ranks).mean(dtype=torch.float32))
    
    
    
    
    

