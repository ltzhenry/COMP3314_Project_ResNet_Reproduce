"""Main entry point for reproducing CIFAR-10 ResNet experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

from datasets.cifar10_loader import get_cifar10_dataloaders
from models import resnet_cifar
from utils.scheduler import build_resnet_cifar_scheduler
from utils.train_eval import EpochMetrics, evaluate, save_checkpoint, train_one_epoch


MODEL_REGISTRY = {
    "PlainNet20": resnet_cifar.PlainNet20,
    "PlainNet32": resnet_cifar.PlainNet32,
    "PlainNet56": resnet_cifar.PlainNet56,
    "PlainNet110": resnet_cifar.PlainNet110,
    "ResNet20": resnet_cifar.ResNet20,
    "ResNet32": resnet_cifar.ResNet32,
    "ResNet56": resnet_cifar.ResNet56,
    "ResNet110": resnet_cifar.ResNet110,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet/PlainNet on CIFAR-10")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="ResNet20")
    parser.add_argument("--epochs", type=int, default=164)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1, dest="lr")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--warmup_epochs", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def select_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        return torch.device("cpu")
    return torch.device(device_str)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint.get("epoch", 0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = select_device(args.device)

    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_fn = MODEL_REGISTRY[args.model]
    model = model_fn().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    steps_per_epoch = len(train_loader)
    scheduler = build_resnet_cifar_scheduler(
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, scaler, args.resume)
        print(f"Resumed from checkpoint {args.resume} at epoch {start_epoch}")

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    history: Dict[str, list] = {"train": [], "test": []}

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
        )
        test_metrics = evaluate(model, test_loader, criterion, device=device, epoch=epoch)

        history["train"].append(train_metrics.__dict__)
        history["test"].append(test_metrics.__dict__)

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
        }
        if scheduler is not None:
            state["scheduler_state"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state"] = scaler.state_dict()

        checkpoint_path = save_checkpoint(
            state,
            checkpoint_dir=str(checkpoints_dir),
            filename=f"{run_name}_epoch{epoch+1}.pth",
        )

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"train_loss={train_metrics.loss:.4f}, train_acc={train_metrics.accuracy:.4f}, "
            f"test_loss={test_metrics.loss:.4f}, test_acc={test_metrics.accuracy:.4f}"
        )
        print(f"Saved checkpoint to {checkpoint_path}")

        log_path = logs_dir / f"{run_name}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
