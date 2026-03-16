import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def count_images_by_class(data_dir: Path):
    class_counts = {}
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        count = len([p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        class_counts[class_dir.name] = count
    return class_counts


def plot_distribution(class_counts, out_path):
    names = list(class_counts.keys())
    values = list(class_counts.values())
    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.title("Class Distribution")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=20)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="EDA for raw NEU-DET")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/figures/raw_class_distribution.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    counts = count_images_by_class(data_dir)
    print("Class counts:")
    for k, v in counts.items():
        print(f"{k}: {v}")
    plot_distribution(counts, args.out)
    print(f"Saved class distribution figure: {args.out}")


if __name__ == "__main__":
    main()
