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
from train import OurModel
import numpy as np

import torch.nn as nn
from transformers import AutoTokenizer, OPTModel

# import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler

from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils


logger = logging.getLogger(__name__)

def val(opt, model, optimizer, scheduler, step):

    # run_stats = utils.WeightedAvgStats()

    # tb_logger = utils.init_tb_logger(opt.output_dir)
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
    val_dataset = data.load_data(opt, tokenizer, val=True)
    logger.warning(f"Data loading finished for rank {dist_utils.get_rank()}")

    train_sampler = RandomSampler(val_dataset)
    train_dataloader = DataLoader(
        val_dataset,
        sampler=train_sampler,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    epoch = 1

    model.eval()
    wandb.init(
      # Set the project where this run will be logged
      project="first-try", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_5_eval", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": opt.lr,
      "batch_size": 1000,
      "min_ratio": 0.3,
      "dataset": "all 8 0.5B training",
      "total_steps": opt.total_steps,
      })
  
    while epoch < 2:
        val_dataset.generate_offset()

        logger.info(f"Start epoch {epoch}")
        
        losses = []
        for i, batch in enumerate(train_dataloader):
            step += 1

            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            # train_loss, iter_stats = model(**batch, stats_prefix="train")

            # TODO: figure out what to do with batch to separate into query, document
            # convert docs, queries into tensors
            docs = torch.zeros(opt.batch_size, vocab_size, dtype=torch.int64, device=device)
            queries = torch.zeros(opt.batch_size, vocab_size, dtype=torch.int64, device=device)
            # docs = docs.cuda()
            # queries = queries.cuda()
            for i, doc in enumerate(batch["k_tokens"]):
                doc_onehot = F.one_hot(doc, vocab_size).sum(0)
                docs[i] = doc_onehot
                query_one_hot = F.one_hot(batch["q_tokens"][i], vocab_size).max(0)[0]
                queries[i] = query_one_hot

            
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
                print(loss.item())

                

                logger.info(log)
                
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
    model_path = os.path.join("experiment4/checkpoint", "step-800")
    print(model_path)
    model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            OurModel,
            model_path,
            opt,
            reset_params=False,
        )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    logger.info(f"Model loaded from {model_path}")
    weights = model.weights
    # print(model.beta)
    # top20 = torch.topk(weights,50)
    # print(top20)
    # indices = top20[1]
    # print(tokenizer.decode(indices))
    
    # bottom20 = torch.topk(weights,50,largest=False)
    # print(bottom20)
    # indices_bot = bottom20[1]
    # print(tokenizer.decode(indices_bot))
    
    val(opt, model, optimizer, scheduler, 0)
    
    
        
    
    
    
    

        
        