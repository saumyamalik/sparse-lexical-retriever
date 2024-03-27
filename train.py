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

class OurModel(nn.Module):
  def __init__(self):
    super().__init__()
    VOCAB_SIZE = 50265
    self.weights = nn.Parameter(torch.rand(VOCAB_SIZE, requires_grad=True))
    self.beta = nn.Parameter(torch.rand(1, requires_grad=True))

  # takes in k-hot encodings of queries, dimension V x B
  # returns B x B vector corresponding to score for each document
  def forward(self, queries, doc_features):
    # make the weights zero'd out for efficiency sake - element wise multiplication
    weights_zeroed = torch.mul(self.weights, queries)

    # apply non-linear function to doc_features
    # func will have dim D x V
    denom = doc_features + torch.exp(self.beta)
    func = torch.divide(doc_features, denom)

    scores = torch.matmul(func, torch.t(weights_zeroed))

    return scores


def train(opt, model, optimizer, scheduler, step):

    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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

    epoch = 1

    model.train()
    wandb.init(
      # Set the project where this run will be logged
      project="sparse-retriever", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_1", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": opt.lr,
      "batch_size": BATCH_SIZE,
      "min_ratio": opt.ratio_min,
      "max_ratio": opt.ratio_max,
      "total_steps": opt.total_steps,
      })
  
    while step < opt.total_steps:
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
                        
            docs_ids = batch['k_tokens']
            queries_ids = batch['q_tokens']

            #docs = F.one_hot(docs_ids, tokenizer.vocab_size).sum(-2)
            docs = torch.zeros(BATCH_SIZE, vocab_size, dtype=docs_ids.dtype, device=docs_ids.device)
            docs = docs.scatter_add_(1, docs_ids, torch.ones_like(docs_ids))

            #for this tokenizer, it is 1
            PAD_TOKEN = 1
            docs[:,PAD_TOKEN] = 0

            # queries = F.one_hot(queries_ids, tokenizer.vocab_size).max(-2)[0]
            queries = torch.zeros(BATCH_SIZE, vocab_size, dtype=queries_ids.dtype, device=queries_ids.device)
            queries = queries.scatter_(1, queries_ids, torch.ones_like(queries_ids))
            #for this tokenizer, it is 1
            PAD_TOKEN = 1
            queries[:,PAD_TOKEN] = 0
                
            logits = model(queries, docs)
            
            train_labels = torch.arange(0,opt.batch_size).cuda()
            loss = F.cross_entropy(logits, train_labels) + F.cross_entropy(torch.t(logits), train_labels)
            predictions = logits.max(1)
            correct = torch.where(predictions[1]==train_labels, 1, 0)
            acc = correct.sum(0) / opt.batch_size

            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            wandb.log({"loss": loss, "acc": acc})
            optimizer.step()

            scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()

            if step % 10 == 0:
                log = f"{step} / {opt.total_steps}"
                # for k, v in sorted(run_stats.average_stats.items()):
                #     log += f" | {k}: {v:.3f}"
                #     if tb_logger:
                #         tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"
                log += f" | loss: {loss.item()}"
                log+= f" | accuracy: {acc}"
                logger.info(log)
                run_stats.reset()

            if step == 1 or step % 160 == 0 or i == len(train_dataloader) - 1:
                utils.save(model, optimizer, scheduler, step, opt, opt.folder_name, f"step-{step}")

            if step > opt.total_steps:
                break
        epoch += 1


def eval_model(opt, query_encoder, doc_encoder, tokenizer, tb_logger, step):
    for datasetname in opt.eval_datasets:
        metrics = beir_utils.evaluate_model(
            query_encoder,
            doc_encoder,
            tokenizer,
            dataset=datasetname,
            batch_size=opt.per_gpu_eval_batch_size,
            norm_doc=opt.norm_doc,
            norm_query=opt.norm_query,
            beir_dir=opt.eval_datasets_dir,
            score_function=opt.score_function,
            lower_case=opt.lower_case,
            normalize_text=opt.eval_normalize_text,
        )

        message = []
        if dist_utils.is_main():
            for metric in ["NDCG@10", "Recall@10", "Recall@100"]:
                message.append(f"{datasetname}/{metric}: {metrics[metric]:.2f}")
                if tb_logger is not None:
                    tb_logger.add_scalar(f"{datasetname}/{metric}", metrics[metric], step)
            logger.info(" | ".join(message))


if __name__ == "__main__":
    logger.info("Start")

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

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
    model_class = OurModel
    if not directory_exists and opt.model_path == "none":
        # CHANGE THIS = done
        model = OurModel()
        model = model.cuda()
        optimizer, scheduler = utils.set_optim(opt, model)
        step = 0
    elif directory_exists:
        model_path = os.path.join(opt.output_dir, "checkpoint", "latest")
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class,
            model_path,
            opt,
            reset_params=False,
        )
        logger.info(f"Model loaded from {opt.output_dir}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class,
            opt.model_path,
            opt,
            reset_params=False if opt.continue_training else True,
        )
        if not opt.continue_training:
            step = 0
        logger.info(f"Model loaded from {opt.model_path}")

    logger.info(utils.get_parameters(model))

    # if dist.is_initialized():
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[opt.local_rank],
    #         output_device=opt.local_rank,
    #         find_unused_parameters=False,
    #     )
    #     dist.barrier()

    logger.info("Start training")
    train(opt, model, optimizer, scheduler, step)
