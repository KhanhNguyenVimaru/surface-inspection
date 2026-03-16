import argparse
import random
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_class_images(image_paths, seed: int):
    train_paths, temp_paths = train_test_split(image_paths, test_size=0.30, random_state=seed, shuffle=True)
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.50, random_state=seed, shuffle=True)
    return train_paths, val_paths, test_paths


def copy_paths(paths, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_path in paths:
        shutil.copy2(src_path, out_dir / src_path.name)


def main():
    parser = argparse.ArgumentParser(description="Split NEU-DET dataset to train/val/test = 70/15/15")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    classes = [p for p in input_dir.iterdir() if p.is_dir()]
    if not classes:
        raise ValueError(f"No class folders found in {input_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    stats = {}
    for class_dir in classes:
        image_paths = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if len(image_paths) < 10:
            raise ValueError(f"Too few images in class '{class_dir.name}'")

        train_paths, val_paths, test_paths = split_class_images(image_paths, args.seed)
        copy_paths(train_paths, output_dir / "train" / class_dir.name)
        copy_paths(val_paths, output_dir / "val" / class_dir.name)
        copy_paths(test_paths, output_dir / "test" / class_dir.name)

        stats[class_dir.name] = {
            "total": len(image_paths),
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
        }

    print("Split completed.")
    for cls, cls_stats in stats.items():
        print(cls, cls_stats)


if __name__ == "__main__":
    main()
