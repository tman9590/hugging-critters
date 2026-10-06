#!/usr/bin/env python3
"""Fine-tune a compact MobileNetV3 classifier on the Eastern-US subset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(class_count: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, class_count)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    seed_everything(args.seed)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(0.25, 0.25, 0.2, 0.03),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = datasets.ImageFolder(args.data / "train", train_transform)
    validation_set = datasets.ImageFolder(args.data / "validation", validation_transform)
    if train_set.classes != validation_set.classes:
        raise RuntimeError("Train/validation class maps differ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(len(train_set.classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=device.type == "cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    labels = {str(i): label for i, label in enumerate(train_set.classes)}
    (args.output / "classmap.json").write_text(json.dumps(labels, indent=2) + "\n")

    best_accuracy = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = seen = 0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels_batch.size(0)
            seen += labels_batch.size(0)
        scheduler.step()
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(json.dumps(row))
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), args.output / "best_model.pth")

    metrics = {
        "architecture": "mobilenet_v3_small",
        "seed": args.seed,
        "device": str(device),
        "classes": train_set.classes,
        "train_images": len(train_set),
        "validation_images": len(validation_set),
        "best_validation_accuracy": best_accuracy,
        "best_epoch": max(history, key=lambda row: row["validation_accuracy"])["epoch"],
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
