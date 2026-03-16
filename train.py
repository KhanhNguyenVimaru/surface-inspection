import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data import class_distribution, load_imagefolder_splits
from src.model import build_model
from src.utils import (
    classification_metrics,
    ensure_dirs,
    evaluate,
    save_confusion_matrix,
    save_curves,
    save_json,
    set_seed,
    train_one_epoch,
)


def main():
    parser = argparse.ArgumentParser(description="Train NEU-DET classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to processed data with train/val/test")
    parser.add_argument("--model", type=str, default="resnet18", choices=["custom_cnn", "resnet18", "mobilenet_v3_small"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = load_imagefolder_splits(args.data_dir)
    class_names = train_ds.classes
    num_classes = len(class_names)
    print("Class distribution (train):", class_distribution(train_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, num_classes=num_classes, pretrained=args.model != "custom_cnn").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ensure_dirs("outputs/checkpoints", "outputs/figures", "outputs/reports")
    best_ckpt_path = Path("outputs/checkpoints") / f"best_{args.model}.pt"

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_name": args.model,
                    "class_names": class_names,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )

    print(f"Best val acc: {best_val_acc:.4f}")
    save_curves(history, f"outputs/figures/curves_{args.model}.png")

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    report_dict, report_text = classification_metrics(y_true, y_pred, class_names)
    save_confusion_matrix(y_true, y_pred, class_names, f"outputs/figures/confusion_{args.model}.png")
    save_json(
        {
            "model": args.model,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "classification_report": report_dict,
        },
        f"outputs/reports/metrics_{args.model}.json",
    )
    print("Test accuracy:", round(test_acc, 4))
    print(report_text)
    print(f"Checkpoint saved: {best_ckpt_path}")


if __name__ == "__main__":
    main()
