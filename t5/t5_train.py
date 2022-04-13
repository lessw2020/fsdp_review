# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from typing import Type

# removed - DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from transformers.models.t5.modeling_t5 import T5Block

# from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from pathlib import Path
from nlp import load_metric
from nlp import load_dataset
from sklearn.model_selection import train_test_split
from wikihow_dataset import *


def transformer_wrapper(
    module: nn.Module,
    recurse: bool,
    unwrapped_params: int,
    transformer_layer_cls: Type[nn.Module],
    min_num_params: int = int(1e8),
) -> bool:

    """policy for wrapping transformers with shared embedding
    shared embeddings will be housed in the outermost layer, thus available to all internal
    fsdp units
    """
    is_large = unwrapped_params >= min_num_params
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return is_large and isinstance(module, transformer_layer_cls)


g_fsdp_unit_params = 1000000

fsdp_wrapping_policy = functools.partial(
    transformer_wrapper,
    min_num_params=g_fsdp_unit_params,
    transformer_layer_cls=T5Block,
)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12369"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    if sampler:
        sampler.set_epoch(epoch)
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(rank)

        """print("************************")
        print(
            "train_loader",
            type(batch),
            batch["source_ids"].size(),
            batch["source_mask"].size(),
            batch["target_ids"].size(),
        )
        print("************************")
        """
        optimizer.zero_grad()
        output = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"],
        )
        # print("##############################")
        # print(output.keys())
        # print("##############################")
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for batch in test_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(rank)
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"],
            )
            ddp_loss[0] += output["loss"].item()  # sum up batch loss
            pred = output.logits.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(batch["target_ids"].view_as(pred)).sum().item()
            ddp_loss[2] += len(batch)

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                int(ddp_loss[1]),
                int(ddp_loss[2]),
                100.0 * ddp_loss[1] / ddp_loss[2],
            )
        )


def ddp_main(rank, world_size, args):

    model_name = "google/t5-v1_1-base"
    printable_model_name = str.replace(model_name, "/", "==")
    # t5-base
    # google/t5-v1_1-small

    #   google/t5-v1_1-base

    #   google/t5-v1_1-large

    #   google/t5-v1_1-xl

    #   google/t5-v1_1-xxl

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n--> {model_name} has {total_params/1e6} Million params\n")
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    dataset = load_dataset("wikihow", "all", data_dir="../Fusion/data/wikihow")
    if rank == 0:
        print(dataset.keys())
        print("Size of train dataset: ", dataset["train"].shape)
        print("Size of Validation dataset: ", dataset["validation"].shape)

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    train_dataset = wikihow(tokenizer, "train", None, 512, 150, True)
    val_dataset = wikihow(tokenizer, "validation", None, 512, 150, True)

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    setup(rank, world_size)

    train_kwargs = {"batch_size": args.batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": args.test_batch_size, "sampler": sampler2}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    # t5 wrap policy
    # param_size_wrapping = 800000
    # t5_wrap_policy = functools.partial(
    #    default_auto_wrap_policy, min_num_params=param_size_wrapping
    # )

    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()

    # model = model.to(rank)
    # model = DDP(model)
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_wrapping_policy,
    ).to(rank)

    if rank == 0:
        print(f"model ")
        fn = printable_model_name + "-shared_layout.txt"
        with open(fn, "w") as external_file:
            header_text = (
                f"model = {model_name}, sharded with {g_fsdp_unit_params} parameters\n"
            )
            print(header_text, file=external_file)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            milli_params = total_params * 4 / 1e6
            print(f"\n--> {model_name} has {milli_params} params\n", file=external_file)
            print(f"model wrapping = \n{model}\n\n", file=external_file)

            external_file.close()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(
            args,
            model,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler1,
        )
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(
            f"Cuda event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        # print(f"{model}")

    if args.save_model:
        dist.barrier()
        states = model.state_dict()
    if rank == 0:
        torch.save(states, "t5_small_wikihow.pt")

    cleanup()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch T5 simple Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0008,
        metavar="LR",
        help="learning rate (default: .0008)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(ddp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
