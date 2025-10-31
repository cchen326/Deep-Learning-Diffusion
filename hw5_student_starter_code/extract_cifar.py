#@title Prepare CIFAR10 image folders for training/evaluation
from pathlib import Path
from torchvision import datasets
from PIL import Image
from tqdm.auto import tqdm

TARGET_SIZE = 32 #@param {type:"integer"}
MAX_TRAIN = None #@param {type:"integer"}
MAX_VAL = None #@param {type:"integer"}

output_root = Path("data/cifar10")
raw_root = Path("data/cifar-10-batches-py")
output_root.mkdir(parents=True, exist_ok=True)

train_ds = datasets.CIFAR10(root=raw_root, train=True, download=True)
val_ds = datasets.CIFAR10(root=raw_root, train=False, download=True)

def _export_split(ds, split_name, limit):
    num_classes = len(ds.classes)
    limit_per_class = None if limit is None else max(1, limit // num_classes)
    per_class_counts = {cls_name: 0 for cls_name in ds.classes}
    total_saved = 0

    for idx in tqdm(range(len(ds)), desc=f"Saving {split_name}"):
        if limit is not None and total_saved >= limit:
            break
        img, label = ds[idx]
        class_name = ds.classes[label]
        if limit_per_class is not None and per_class_counts[class_name] >= limit_per_class:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        if TARGET_SIZE != img.size[0]:
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
        class_dir = output_root / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        save_idx = per_class_counts[class_name]
        img.save(class_dir / f"{save_idx:05d}.png")
        per_class_counts[class_name] += 1
        total_saved += 1

_export_split(train_ds, "train", MAX_TRAIN)
_export_split(val_ds, "val", MAX_VAL)
print("Dataset ready at", output_root.resolve())