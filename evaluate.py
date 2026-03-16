import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import load_imagefolder_splits
from src.model import build_model
from src.utils import classification_metrics, ensure_dirs, evaluate, save_confusion_matrix, save_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["custom_cnn", "resnet18", "mobilenet_v3_small"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_ds = load_imagefolder_splits(args.data_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(test_ds.classes)
    model = build_model(args.model, num_classes=num_classes, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    report_dict, report_text = classification_metrics(y_true, y_pred, test_ds.classes)

    ensure_dirs("outputs/figures", "outputs/reports")
    ckpt_name = Path(args.checkpoint).stem
    save_confusion_matrix(y_true, y_pred, test_ds.classes, f"outputs/figures/confusion_{ckpt_name}.png")
    save_json(
        {
            "checkpoint": args.checkpoint,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "classification_report": report_dict,
        },
        f"outputs/reports/eval_{ckpt_name}.json",
    )

    print("Test loss:", round(test_loss, 4))
    print("Test accuracy:", round(test_acc, 4))
    print(report_text)


if __name__ == "__main__":
    main()
