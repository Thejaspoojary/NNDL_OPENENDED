# Dataset Download Instructions

## Option 1: Manual download from Kaggle (recommended for first-time users)

1. Create a Kaggle account (https://www.kaggle.com) if you do not have one.
2. Go to the dataset page:
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
3. Click "Download" and wait for the archive to be downloaded.
4. Extract the archive. The dataset contains folders like `chest_xray/`, or
   top-level `train/` and `test/` style folders depending on the version.
5. Place the extracted chest X-ray folders into the project structure:

```
Pneumonia-Detection/
└── dataset/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── test/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test_data/
        ├── README.txt
        └── .gitkeep
```

6. Confirm the folders contain image files (e.g. `.jpg`, `.jpeg`, `.png`).

## Option 2: Kaggle API download (optional)

```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
kaggle datasets unzip -d dataset/
```

You must be logged in via the Kaggle API (`kaggle` config) with your own
account. Never commit your credentials.

## Notes

- The project never generates or substitutes fake images.
- The official Kaggle test split should be preserved.
- See the main README.md for the full workflow.
