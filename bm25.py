# Compute the BM25 IDF statistics for the training corpus
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
from transformers import AutoTokenizer, OPTModel

# import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler

from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils
# from src import moco, inbatch


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Start")

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    # slurm.init_distributed_mode(opt)
    # slurm.init_signal_handler()

    directory_exists = os.path.isdir(opt.output_dir)
    # if dist.is_initialized():
    #     dist.barrier()
    # os.makedirs(opt.output_dir, exist_ok=True)
    # if not directory_exists and dist_utils.is_main():
    #     options.print_options(opt)
    # if dist.is_initialized():
    #     dist.barrier()
    utils.init_logger(opt)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # if opt.contrastive_mode == "moco":
    #     model_class = moco.MoCo
    # elif opt.contrastive_mode == "inbatch":
    #     model_class = inbatch.InBatch
    # else:
    #     raise ValueError(f"contrastive mode: {opt.contrastive_mode} not recognised")

    # if dist.is_initialized():
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[opt.local_rank],
    #         output_device=opt.local_rank,
    #         find_unused_parameters=False,
    #     )
    #     dist.barrier()

    logger.info("Start training")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    vocab_size = tokenizer.vocab_size

    logger.info("Data loading")
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     tokenizer = model.module.tokenizer
    # else:
    #     tokenizer = model.tokenizer
    collator = data.Collator(opt=opt)
    
    BATCH_SIZE = opt.batch_size
    train_dataset = data.load_data(opt, tokenizer)
    logger.warning(f"Data loading finished for rank {dist_utils.get_rank()}")
    
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    epoch = 1
    step=0
    
    bm25_weights = torch.zeros(vocab_size, device="cuda:0")
    
    while epoch < 2:
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=opt.batch_size,
            drop_last=True,
            num_workers=opt.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )
        
        train_dataset.generate_offset()

        logger.info(f"Start epoch {epoch}")
        
        losses = []
        for i, batch in enumerate(train_dataloader):
            step += 1

            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            
            # Only care about documents
            docs_ids = batch['k_tokens']
            # queries_ids = batch['q_tokens']


            docs = torch.zeros(BATCH_SIZE, vocab_size, dtype=docs_ids.dtype, device=docs_ids.device)
            # NOT scatter add bc we only care about presence, not count
            docs = docs.scatter_(1, docs_ids, torch.ones_like(docs_ids))
            docs = torch.sum(docs, 0)
            
            bm25_weights = bm25_weights + docs
            
            if step % 10 == 0:
                log = f"{step} / {opt.total_steps}"
            
        epoch += 1
    
    # do some computation on these weights
    #for this tokenizer, it is 1
    PAD_TOKEN = 1
    bm25_weights[PAD_TOKEN] = 0
    N = len(train_dataset)
    bm25_weights = torch.divide((N - bm25_weights + 0.5), bm25_weights + 0.5) + 1
    bm25_weights = torch.log(bm25_weights)
    
    fp = './bm25/weights'
    checkpoint = {
        "IDF": bm25_weights
    }
    torch.save(checkpoint, fp)
