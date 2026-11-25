"""Dataset loading, validation and train/val/test handling."""

from pathlib import Path

import numpy as np

from src import config
from src.preprocessing import IMAGE_PATTERNS, image_to_array, load_image


def list_image_files(class_dir: Path):
    """Return a sorted list of image file paths inside a class folder."""
    class_dir = Path(class_dir)
    if not class_dir.is_dir():
        return []
    files = []
    for pattern in IMAGE_PATTERNS:
        for p in class_dir.glob(pattern):
            if p.is_file():
                files.append(p)
    return sorted(files, key=lambda p: str(p))


def get_split_counts(split_dir: Path) -> dict:
    """Return per-class image counts and totals for a split directory."""
    split_dir = Path(split_dir)
    counts = {}
    total = 0
    for class_name in config.CLASS_NAMES:
        files = list_image_files(split_dir / class_name)
        counts[class_name] = len(files)
        total += len(files)
    return {"counts": counts, "total": total, "exists": split_dir.is_dir()}


def validate_dataset(data_root=None, splits=("train", "val", "test")) -> dict:
    """Validate the chest X-ray dataset layout.

    Returns a dictionary describing whether the dataset is usable and the
    number of images found per split and class.
    """
    data_root = Path(data_root or config.DATA_DIR)
    split_results = {}
    for split in splits:
        split_dir = data_root / split
        counts = get_split_counts(split_dir)
        ok = counts["exists"] and all(
            counts["counts"].get(c, 0) > 0 for c in config.CLASS_NAMES
        )
        split_results[split] = {
            "path": str(split_dir),
            "exists": bool(counts["exists"]),
            "valid": bool(ok),
            "counts": counts["counts"],
            "total": int(counts["total"]),
        }

    train = split_results.get("train", {})
    train_valid = bool(train.get("valid"))
    val = split_results.get("val", {})
    test = split_results.get("test", {})

    other_splits_valid = all(
        bool(split_results[s].get("valid")) for s in ("val", "test") if split_results.get(s, {}).get("exists")
    )
    valid = bool(train_valid and other_splits_valid)

    if not train_valid:
        if not train.get("exists"):
            message = (
                "Training data not found. Please download the Chest X-Ray Images "
                "(Pneumonia) dataset from Kaggle and place it inside the 'dataset/train' "
                "folder with 'NORMAL' and 'PNEUMONIA' subfolders. See README.md for instructions."
            )
        else:
            missing = [c for c in config.CLASS_NAMES if train["counts"].get(c, 0) == 0]
            message = (
                f"Training data is incomplete. Missing or empty class folder(s): {', '.join(missing)}. "
                "Please re-check the dataset download and folder structure."
            )
    else:
        message = "Dataset is valid."

    return {
        "valid": bool(valid),
        "message": message,
        "splits": split_results,
        "class_names": list(config.CLASS_NAMES),
    }


def create_validation_split(data_root=None, train_ratio: float = 0.2, seed: int = config.RANDOM_SEED) -> Path:
    """Create a validation split from the training data (fixed random seed).

    Only used when the official validation split is not available. The split
    is performed from the training images only, using a fixed random seed.
    """
    import shutil

    data_root = Path(data_root or config.DATA_DIR)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    all_files = []
    labels = []
    for class_name in config.CLASS_NAMES:
        for p in list_image_files(train_dir / class_name):
            all_files.append(p)
            labels.append(config.CLASS_NAMES.index(class_name))

    rng = np.random.RandomState(seed)
    order = list(range(len(all_files)))
    rng.shuffle(order)
    n_val = max(1, int(len(order) * train_ratio))
    val_set = set(order[:n_val])
    val_files = [all_files[i] for i in sorted(val_set)]
    val_names = {p.name for p in val_files}

    for class_name in config.CLASS_NAMES:
        target_dir = val_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for p in list_image_files(train_dir / class_name):
            if p.name in val_names and p not in val_files:
                continue
            if p.name in val_names and p in val_files:
                dest = target_dir / p.name
                if not dest.exists():
                    shutil.copy(str(p), str(dest))
    return val_dir


def compute_class_weights(y_train) -> dict:
    """Compute balanced class weights from training label counts.

    Returns a mapping of class index (0 = NORMAL, 1 = PNEUMONIA) to weight, as
    expected by Keras ``class_weight``.
    """
    y_train = np.asarray(y_train)
    counts = np.bincount(y_train.astype(int), minlength=len(config.CLASS_NAMES))
    total = float(counts.sum()) if counts.sum() > 0 else 1.0
    weights = {}
    for idx, class_name in enumerate(config.CLASS_NAMES):
        count = float(counts[idx]) if idx < len(counts) else 0.0
        weights[int(idx)] = float(total / (len(config.CLASS_NAMES) * max(count, 1e-6)))
    return weights


def augment_image(image):
    """Small random rotations, flips and zoom for training augmentation."""
    import tensorflow as tf

    image = tf.image.flip_left_right(image)
    angle = tf.random.uniform([], -0.06, 0.06)
    image = tf.image.rotate(image, angle, interpolation="bilinear")
    zoom = tf.random.uniform([], 1.0, 1.15)
    new_size = max(8, int(224 * float(zoom)))
    image = tf.image.resize_with_crop_or_pad(image, new_size, new_size)
    return image


def build_tf_dataset(x, y, batch_size=config.BATCH_SIZE, shuffle=False, augment=False, seed=config.RANDOM_SEED):
    """Build a tf.data dataset from numpy arrays."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(int(ds.cardinality().numpy()), seed=seed, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(lambda img, lbl: (augment_image(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda img, lbl: (img, lbl))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


_SPLIT_DIRS = {
    "train": config.TRAIN_DIR,
    "val": config.VALIDATION_DIR,
    "test": config.TEST_DIR,
}


def load_split(split: str, img_size=tuple(config.IMG_SIZE), model_name: str = "cnn"):
    """Load all images of a split into numpy arrays.

    Returns ``(x, y, file_paths)``. ``x`` has shape ``(n, h, w, 3)`` float32 in
    [0, 1]; ``y`` holds integer class labels. Undecodable files are skipped.
    """
    from src.preprocessing import preprocess_image

    split_dir = Path(_SPLIT_DIRS.get(split, config.DATA_DIR / split))
    x_list = []
    y_list = []
    paths = []
    for class_name in config.CLASS_NAMES:
        label = config.CLASS_NAMES.index(class_name)
        for p in list_image_files(split_dir / class_name):
            try:
                arr = preprocess_image(image_to_array(load_image(p)), model_name)
                arr = np.asarray(arr, dtype=np.float32).reshape((int(config.IMG_SIZE[0]), int(config.IMG_SIZE[1]), 3))
                x_list.append(arr)
                y_list.append(label)
                paths.append(str(p))
            except Exception:
                continue
    if not x_list:
        return (
            np.zeros((0, int(config.IMG_SIZE[0]), int(config.IMG_SIZE[1]), 3), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            [],
        )
    x = np.stack(x_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    return x, y, paths
