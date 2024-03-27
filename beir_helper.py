import sys
import argparse
import torch
import logging
import json
import numpy as np
import os

import src.slurm
import src.contriever
import src.beir_utils
import src.utils
import src.dist_utils
import src.contriever


from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils
from train import OurModel
from transformers import AutoTokenizer, OPTModel
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def load_model(model_name_or_path, step_path):
    options = Options()
    opt = options.parse()
    #model_path = os.path.join("experiment4/checkpoint", "step-800")
    model_path = os.path.join(model_name_or_path, step_path)
    print(model_path)
    model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            OurModel,
            model_path,
            opt,
            reset_params=False,
        )
    logger.info(f"Model loaded from {model_path}")
    return model, optimizer, scheduler

# applies weights to queries (where queries exist)
class QueryEncoder(nn.Module):
    def __init__(self, weights, tokenizer):
        super().__init__()
        self.weights = weights
        self.tokenizer = tokenizer

    # applies weights to queries where queries exist
    def forward(self, **kwargs):
        query_tokens = kwargs['input_ids']
        # query_tokens = batch_encoded['input_ids']
        # attn_mask = batch_encoded['attention_mask']
       
        #queries = torch.tensor(query_tokens, dtype=torch.int64)
        queries = query_tokens
        query_hot = F.one_hot(queries, self.tokenizer.vocab_size).max(-2)[0]
        #for this tokenizer, it is 1
        PAD_TOKEN = 1
        query_hot[:,1] = 0
        
        encoded = torch.mul(self.weights, query_hot)
        return encoded
    

class DocEncoder(nn.Module):
    def __init__(self, beta, tokenizer, isBM25):
        super().__init__()
        self.beta = beta
        self.tokenizer = tokenizer
        self.is_BM25 = isBM25

    # applies non-linearity to doc features where queries exist
    def forward(self, **kwargs):
        doc_tokens = kwargs['input_ids']
        #attn_mask = batch_encoded['attention_mask']
       
        #docs = torch.tensor(doc_tokens, dtype=torch.int64)
        docs = doc_tokens

        doc_features = F.one_hot(docs, self.tokenizer.vocab_size).sum(-2)
        #for this tokenizer, it is 1
        PAD_TOKEN = 1
        doc_features[:,1] = 0
        
        if self.is_BM25:
            doc_lengths = torch.sum(doc_features, 1)
            avgdl = doc_lengths.mean(dtype=torch.float32)

            d_avg = doc_lengths/avgdl
            k = 1.6
            b = 0.75
            denom = doc_features + torch.t(torch.unsqueeze(k * (1-b + b * d_avg), 0))
            num = doc_features * (k + 1)
            encoded = torch.divide(num, denom)
            
        else:
            denom = doc_features + torch.exp(self.beta)
            encoded = torch.divide(doc_features, denom)

        return encoded
  
def load_retriever(model_name_or_path, step_path, isBM25=False):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if isBM25:
        weights = torch.load('./bm25/weights')['IDF']
        # assign a dummy value
        nonlinearity=0
    
    else:
        model, optimizer, scheduler = load_model(model_name_or_path, step_path)
        
        weights = model.weights
        nonlinearity = model.beta
    
    
    query_encoder = QueryEncoder(weights, tokenizer)
    doc_encoder = DocEncoder(nonlinearity, tokenizer, isBM25)
    
    return tokenizer, query_encoder, doc_encoder
    
    
    
    
    
    
    
