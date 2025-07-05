import os
from os.path import exists, join, isdir
import gc
import json
import math
import random
import copy
from copy import deepcopy
from tqdm import tqdm
import zipfile

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Callable, List, Tuple, Union, Any

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
from transformers import TrainingArguments, Trainer, set_seed

from data_helper import CustomDataCollator, CustomDataset, GraphDataset, VerilogGraphDataset
from aigmae.configuration_vgmae import AIGMAEConfig
from aigmae.modeling_vgmae import AIGMAEModel

from custom_trainer import CustomTrainer

import warnings

warnings.filterwarnings("ignore")

@dataclass
class DataArguments:
    root_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    pyg_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    verilog_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: str = field(default="none")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(
        default="adamw_torch"
    )  # "adamw_torch"
    lr_scheduler_type: str = field(
        default="cosine"
    )  # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"
    min_lr_ratio: float = field(
        default=0.1
    )
    lr_scheduler_kwargs: Dict[str, str] = field(
        default_factory=lambda:{"num_cycles": 0.5}
    )


def train():
    parser = transformers.HfArgumentParser(
        (DataArguments, TrainingArguments)
    )
    data_args, training_args = parser.parse_args_into_dataclasses()
    num_cycles = math.acos(training_args.min_lr_ratio * 2 - 1) / (math.pi * 2)
    training_args.lr_scheduler_kwargs["num_cycles"] = num_cycles
    # print(training_args.lr_scheduler_kwargs)

    mgm_dataset = GraphDataset(root_path=data_args.root_path, data_path=data_args.data_path)
    vg_dataset = VerilogGraphDataset(pyg_path=data_args.pyg_path, verilog_path=data_args.verilog_path)
    train_dataset = CustomDataset([mgm_dataset, vg_dataset])

    model_config = AIGMAEConfig(
        num_classes = 4,
        num_encoder_layers = 7,
        num_cross_decoder_layers = 2,
        hidden_size = 64,
        cross_hidden_size = 3584,
        cross_num_heads = 8,
    )
    mgm_mask_ratio = 0.3
    align_mask_ratio = 0.7
    cross_hidden_size = 3584

    model = AIGMAEModel(model_config)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)
    scheduler = None

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CustomDataCollator(mgm_mask_ratio=mgm_mask_ratio, 
                                         align_mask_ratio=align_mask_ratio, 
                                         cross_hidden_size=cross_hidden_size),
        optimizers=(optimizer, scheduler),
        extra_losses=["node_loss", "vg_node_loss", "indegree_loss", "outdegree_loss", "indegree_loss_t", "outdegree_loss_t"]
    )

    trainer.train()
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    set_seed(42)
    train()