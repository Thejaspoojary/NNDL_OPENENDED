"""Training pipeline for the custom CNN and the MobileNetV2 transfer model."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src import config
from src.cnn_model import build_cnn, compile_cnn
from src.data_loader import (
    build_tf_dataset,
    compute_class_weights,
    create_validation_split,
    load_split,
    validate_dataset,
)
from src.transfer_learning import build_mobilenetv2, compile_mobilenetv2


def save_history(history, model_name: str) -> dict:
    """Save training history as JSON and CSV, return the plain dict."""
    history_dir = config.HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, values in history.history.items():
        payload[key] = [float(v) for v in values]
    with open(history_dir / f"{model_name}_history.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(history_dir / f"{model_name}_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])
        for i in range(len(history.history.get("loss", []))):
            row = [i + 1]
            row.append(float(history.history["loss"][i]))
            row.append(float(history.history["accuracy"][i]) if "accuracy" in history.history else float("nan"))
            row.append(float(history.history["val_loss"][i]) if "val_loss" in history.history else float("nan"))
            row.append(float(history.history["val_accuracy"][i]) if "val_accuracy" in history.history else float("nan"))
            writer.writerow(row)
    return payload


def save_training_curves(history, model_name: str) -> str:
    """Save accuracy/loss curves plot. Returns the saved filename."""
    plots_dir = config.PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get("loss", []), label="Training loss")
    axes[0].plot(history.history.get("val_loss", []), label="Validation loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history.get("accuracy", []), label="Training accuracy")
    axes[1].plot(history.history.get("val_accuracy", []), label="Validation accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    out = plots_dir / f"{model_name}_training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def _prepare_data(data_root=None, model_name: str = "cnn"):
    validation = validate_dataset(data_root)
    if not validation["valid"]:
        raise FileNotFoundError(
            f"Dataset validation failed: {validation.get('message')} "
            f"Details: {validation.get('splits', {})}"
        )
    x_train, y_train, _ = load_split("train", model_name=model_name)
    if len(x_train) == 0:
        raise ValueError("No valid training images were found. Aborting training.")
    x_val, y_val, _ = load_split("val", model_name=model_name)
    if len(x_val) == 0:
        create_validation_split(data_root=data_root)
        x_val, y_val, _ = load_split("val", model_name=model_name)
    return x_train, y_train, x_val, y_val, validation


def _run_fit(model, checkpoint_path: str, x_train, y_train, x_val, y_val, epochs: int, batch_size: int) -> tf.keras.callbacks.History:
    return model.fit(
        build_tf_dataset(x_train, y_train, batch_size=batch_size, shuffle=True, augment=True),
        validation_data=build_tf_dataset(x_val, y_val, batch_size=batch_size, shuffle=False, augment=False),
        epochs=epochs,
        class_weight=compute_class_weights(y_train),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=config.REDUCE_LR_PATIENCE, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True),
        ],
    )


def train_cnn(epochs: int, batch_size: int, data_root=None, learning_rate: float = config.LEARNING_RATE) -> dict:
    """Train the custom CNN on the real chest X-ray dataset."""
    x_train, y_train, x_val, y_val, validation = _prepare_data(data_root=data_root, model_name="cnn")
    model = build_cnn()
    compile_cnn(model, learning_rate=learning_rate)
    history = _run_fit(model, str(config.CNN_MODEL_PATH), x_train, y_train, x_val, y_val, epochs, batch_size)
    history_payload = save_history(history, "cnn")
    save_training_curves(history, "cnn")
    model.save(str(config.CNN_MODEL_PATH))
    return {"model": model, "history": history_payload, "validation": validation, "best_model_path": str(config.CNN_MODEL_PATH)}


def train_mobilenetv2(epochs: int, batch_size: int, data_root=None, fine_tune: bool = False) -> dict:
    """Train the MobileNetV2 transfer learning model on the real dataset."""
    x_train, y_train, x_val, y_val, validation = _prepare_data(data_root=data_root, model_name="mobilenetv2")
    model = build_mobilenetv2(fine_tune=fine_tune)
    compile_mobilenetv2(model, learning_rate=config.FINE_TUNE_LEARNING_RATE if fine_tune else config.LEARNING_RATE)
    history = _run_fit(model, str(config.MOBILENETV2_MODEL_PATH), x_train, y_train, x_val, y_val, epochs, batch_size)
    history_payload = save_history(history, "mobilenetv2")
    save_training_curves(history, "mobilenetv2")
    model.save(str(config.MOBILENETV2_MODEL_PATH))
    return {"model": model, "history": history_payload, "validation": validation, "best_model_path": str(config.MOBILENETV2_MODEL_PATH)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train pneumonia detection models (CNN or MobileNetV2).")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mobilenetv2"], help="Model to train: 'cnn' or 'mobilenetv2'")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--fine-tune", action="store_true", help="Fine tune the MobileNetV2 backbone (transfer learning only)")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional dataset root directory")
    args = parser.parse_args(argv)

    if args.model == "cnn":
        result = train_cnn(epochs=args.epochs, batch_size=args.batch_size, data_root=args.data_dir)
        best_path = str(config.CNN_MODEL_PATH)
    else:
        result = train_mobilenetv2(epochs=args.epochs, batch_size=args.batch_size, data_root=args.data_dir, fine_tune=args.fine_tune)
        best_path = str(config.MOBILENETV2_MODEL_PATH)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "training_summary.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "best_model_path": best_path,
                "epochs_requested": args.epochs,
                "batch_size": args.batch_size,
                "fine_tune": bool(args.fine_tune),
            },
            f,
            indent=2,
        )
    with open(config.RESULTS_DIR / "preprocessing_config.json", "w") as f:
        json.dump(
            {
                "image_size": [int(config.IMG_SIZE[0]), int(config.IMG_SIZE[1])],
                "class_names": list(config.CLASS_NAMES),
                "normalization": "float32 in [0,1] for custom CNN; mobilenetv2.preprocess_input for MobileNetV2",
                "models": {
                    "cnn": {"input_size": [int(config.IMG_SIZE[0]), int(config.IMG_SIZE[1]), 3]},
                    "mobilenetv2": {"input_size": [int(config.IMG_SIZE[0]), int(config.IMG_SIZE[1]), 3]},
                },
            },
            f,
            indent=2,
        )
    print(f"Training completed for model '{args.model}'. Best model saved to: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
