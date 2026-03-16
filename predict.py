import argparse

import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from src.config import IMG_SIZE
from src.model import build_model


def preprocess_surface_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    denoised = gray.filter(ImageFilter.MedianFilter(size=3))
    enhanced = ImageOps.autocontrast(denoised, cutoff=1)
    return enhanced.convert("RGB")


def infer_one_image(image_path: str, checkpoint_path: str, model_name: str, apply_filter: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_model(model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path)
    if apply_filter:
        image = preprocess_surface_image(image)
    else:
        image = image.convert("RGB")

    tensor = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    pred_label = class_names[pred_idx.item()]
    return pred_label, conf.item()


def main():
    parser = argparse.ArgumentParser(description="Predict one image with trained model")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=["custom_cnn", "resnet18", "mobilenet_v3_small"])
    parser.add_argument("--apply-filter", action="store_true", help="Apply grayscale + denoise + autocontrast filter")
    args = parser.parse_args()

    pred, conf = infer_one_image(args.image, args.checkpoint, args.model, apply_filter=args.apply_filter)
    print(f"Prediction: {pred}")
    print(f"Confidence: {conf:.4f}")


if __name__ == "__main__":
    main()
