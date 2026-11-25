"""Flask web application for pneumonia detection from chest X-ray images."""

import json

from flask import Flask, jsonify, render_template, request, send_from_directory

from src import config
from src.data_loader import validate_dataset
from src.predict import model_exists, predict_uploaded


app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _read_json(path, default=None):
    import os

    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _recommended_model() -> str:
    info = _read_json(config.BEST_MODEL_INFO_PATH, {}) or {}
    best = info.get("best_model") if isinstance(info, dict) else None
    if best in ("cnn", "mobilenetv2"):
        return best
    return config.DEFAULT_MODEL


@app.route("/")
def index():
    dataset_info = validate_dataset()
    splits = dataset_info.get("splits", {}) if isinstance(dataset_info, dict) else {}
    models = {name: model_exists(name) for name in ("cnn", "mobilenetv2")}
    evaluation = _read_json(config.EVALUATION_METRICS_PATH, {}) or {}
    return render_template(
        "index.html",
        dataset_info=dataset_info,
        splits=splits,
        models=models,
        recommended_model=_recommended_model(),
        evaluation=evaluation,
    )


@app.route("/predict")
def predict():
    return render_template(
        "predict.html",
        models={name: model_exists(name) for name in ("cnn", "mobilenetv2")},
        recommended_model=_recommended_model(),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    model_name = (request.form.get("model") or _recommended_model() or config.DEFAULT_MODEL).lower()
    if model_name not in ("cnn", "mobilenetv2"):
        return jsonify(ok=False, error=f"Unknown model '{model_name}'. Choose one of: cnn, mobilenetv2."), 400

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(ok=False, error="No image file was uploaded. Please choose a chest X-ray image."), 400

    result, temp_path = predict_uploaded(file, model_name=model_name)
    if temp_path is not None and hasattr(temp_path, "exists") and temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass
    if not result.get("ok"):
        return jsonify(ok=False, error=result.get("error", "Prediction failed."), model=model_name), int(result.get("status") or 400)
    payload = dict(result)
    payload["ok"] = True
    return jsonify(payload)


@app.route("/compare")
def comparison():
    evaluation = _read_json(config.EVALUATION_METRICS_PATH, {}) or {}
    return render_template(
        "comparison.html",
        evaluation=evaluation,
        recommended_model=_recommended_model(),
    )


@app.route("/dataset")
def dataset():
    dataset_info = validate_dataset()
    return render_template("dataset.html", dataset_info=dataset_info)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "models": {name: model_exists(name) for name in ("cnn", "mobilenetv2")},
            "recommended_model": _recommended_model(),
            "dataset": validate_dataset(),
        }
    )


@app.route("/results/<path:filename>")
def results_file(filename: str):
    plots_dir = config.PLOTS_DIR
    return send_from_directory(str(plots_dir), filename, mimetype="image/png")


@app.errorhandler(413)
def too_large(_e):
    return jsonify(ok=False, error="Uploaded image is too large. Maximum allowed size is 5 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
