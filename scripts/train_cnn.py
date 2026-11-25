"""Convenience script: train the custom CNN model."""

from src.train import main

if __name__ == "__main__":
    raise SystemExit(main(["--model", "cnn"]))
