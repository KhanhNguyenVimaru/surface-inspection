import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from predict import infer_one_image


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
UPLOAD_DIR = Path("uploads")
DEFAULT_CHECKPOINT = "outputs/checkpoints/best_resnet18.pt"
DEFAULT_MODEL = "resnet18"

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "service": "surface-defect-api",
            "message": "Use /api/health and /api/predict",
        }
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "checkpoint_exists": Path(DEFAULT_CHECKPOINT).exists(),
            "checkpoint_path": DEFAULT_CHECKPOINT,
            "model": DEFAULT_MODEL,
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict_image():
    checkpoint_exists = Path(DEFAULT_CHECKPOINT).exists()
    if not checkpoint_exists:
        return (
            jsonify(
                {
                    "error": "checkpoint_not_found",
                    "message": f"Checkpoint not found at: {DEFAULT_CHECKPOINT}. Train model first.",
                }
            ),
            400,
        )

    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return jsonify({"error": "missing_image", "message": "Please choose an image file."}), 400

    if not is_allowed_file(image_file.filename):
        return (
            jsonify(
                {
                    "error": "unsupported_type",
                    "message": "Unsupported file type. Use: .jpg, .jpeg, .png, .bmp",
                }
            ),
            400,
        )

    safe_name = secure_filename(image_file.filename)
    file_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = UPLOAD_DIR / file_name
    image_file.save(file_path)
    apply_filter = parse_bool(request.form.get("apply_filter"), default=True)

    pred_label, confidence = infer_one_image(
        image_path=str(file_path),
        checkpoint_path=DEFAULT_CHECKPOINT,
        model_name=DEFAULT_MODEL,
        apply_filter=apply_filter,
    )

    return jsonify(
        {
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),
            "model_name": DEFAULT_MODEL,
            "uploaded_file": file_name,
            "preprocessing": {
                "apply_filter": apply_filter,
                "pipeline": "grayscale -> median_filter(3x3) -> autocontrast(cutoff=1)",
            },
        }
    )


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
