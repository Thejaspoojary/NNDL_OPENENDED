"""Convenience script: train the MobileNetV2 transfer learning model."""

from src.train import main

if __name__ == "__main__":
    raise SystemExit(main(["--model", "mobilenetv2"]))
