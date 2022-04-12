# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# remove DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from transformers import AutoTokenizer

import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.fully_sharded_data_parallel import (
##    FullyShardedDataParallel as FSDP,
#    CPUOffload,
#    BackwardPrefetch,
# )
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from pathlib import Path

from sklearn.model_selection import train_test_split
import os

# os.environ["TORCH_CPP_SHOW_STACKTRACES="] = "1"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12368"

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

        optimizer.zero_grad()
        output = model(**batch)
        # print("************************")
        # print("************************")
        # print("train_loder",type(output), output)
        # print("************************")
        loss = F.nll_loss(output["logits"], batch["labels"], reduction="sum")
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("\n")
        print(
            "\nTrain Epoch: {} \tLoss: {:.6f}\n".format(
                epoch, ddp_loss[0] / ddp_loss[1]
            )
        )


def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for batch in test_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(rank)
            output = model(**batch)
            ddp_loss[0] += F.nll_loss(
                output["logits"], batch["labels"], reduction="sum"
            ).item()  # sum up batch loss
            pred = output["logits"].argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(batch["labels"].view_as(pred)).sum().item()
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


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir == "neg" else 1)

    return texts, labels


def ddp_main(rank, world_size, args):

    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)

    model_name = "bert-base-uncased"

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if rank == 0:
        print(f"--> Training for {model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n--> {model_name} has {total_params} params\n")

    train_texts, train_labels = read_imdb_split("aclImdb/train")
    test_texts, test_labels = read_imdb_split("aclImdb/test")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2
    )
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)

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

    bert_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=50000)

    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()

    # model = model.to(rank)
    fpSixteen = MixedPrecision(
        # param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )

    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    """  module: nn.Module,
        process_group: Optional[ProcessGroup] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[Callable] = None,
        backward_prefetch: Optional[BackwardPrefetch] = None,
        mixed_precision: Optional[MixedPrecision] = None
    ):"""

    model = FSDP(
        model,
        auto_wrap_policy=bert_wrap_policy,
        mixed_precision=bfSixteen,
    ).to(rank)

    if rank == 0:
        print(f"model via static: {FSDP.fsdp_modules(model, root_only=True)} ")

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
        print(f"{model}")

    if args.save_model:
        dist.barrier()
        states = model.state_dict()
    if rank == 0:
        torch.save(states, "mnist_cnn.pt")

    cleanup()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
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
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
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
