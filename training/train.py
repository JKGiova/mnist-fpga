from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistMLP


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "mnist-mlp-fp32.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 784-16-10 MNIST MLP."
    )

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    # ToTensor converts MNIST pixels from uint8 [0, 255]
    # to float32 [0.0, 1.0].
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    device = choose_device()
    print(f"Device: {device}")

    train_loader, test_loader = create_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = MnistMLP().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_accuracy = 0.0
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train accuracy {train_accuracy:.2f}% | "
            f"test loss {test_loss:.4f} | "
            f"test accuracy {test_accuracy:.2f}%"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("Training completed without producing a checkpoint.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": best_state_dict,
        "architecture": {
            "input_features": 784,
            "hidden_features": 16,
            "output_features": 10,
            "activation": "ReLU",
        },
        "input_preprocessing": {
            "source": "MNIST uint8 pixels",
            "operation": "pixel / 255.0",
            "range": [0.0, 1.0],
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "best_test_accuracy": best_accuracy,
    }

    torch.save(checkpoint, args.output)

    print()
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()