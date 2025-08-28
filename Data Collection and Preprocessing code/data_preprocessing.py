import cv2
import os
from glob import glob
import math
import random
import numpy as np
import shutil
from typing import List


# Config
ROOT = "data_preprocessing"  # your main folder (processed images live here)
IMG_SIZE = 224               # final square size
N_AUG_PER_IMAGE = 2          # how many augmented copies per image

# Geometry
SCALE_RANGE = (0.8, 1.0)     # for RandomResizedCrop (fraction of area)
RATIO_RANGE = (0.9, 1.1)     # slight aspect ratio jitter
HFLIP_PROB = 0.5
ROT_MAX_DEG = 10             # ± degrees

# Photometric
BRIGHTNESS = 0.20            # +/- 20% of 255 -> beta in [-51, +51]
CONTRAST = 0.20            # multiply alpha in [0.8, 1.2]
SATURATION = 0.15            # sat factor in [0.85, 1.15]
HUE = 0.02            # hue shift fraction of 180 -> ±3.6 deg
GRAYSCALE_PROB = 0.15
BLUR_PROB = 0.30
exts = ["*.jpg", "*.jpeg", "*.png"]


def list_images(root: str) -> List[str]:
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(files)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def split_by_class(src_root: str, split_root: str,
                   train_ratio: float, val_ratio: float, seed: int = 42):
    random.seed(seed)
    all_imgs = list_images(src_root)

    # Group by their immediate folder to split per-folder
    folders = {}
    for img_path in all_imgs:
        rel_dir = os.path.dirname(os.path.relpath(img_path, src_root))
        folders.setdefault(rel_dir, []).append(img_path)

    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(split_root, split))

    for rel_dir, paths in folders.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        split_sets = {
            "train": paths[:n_train],
            "val":   paths[n_train:n_train+n_val],
            "test":  paths[n_train+n_val:]
        }

        for split, items in split_sets.items():
            for src_path in items:
                dst_path = os.path.join(split_root, split, rel_dir, os.path.basename(src_path))
                ensure_dir(os.path.dirname(dst_path))
                shutil.copy2(src_path, dst_path)

        print(f"[SPLIT] {rel_dir}: train={n_train}, val={n_val}, test={n_test}")

def preprocessing():
    # Main paths
    source_root = "split_images"           # main dataset folder
    target_root = "data_preprocessing"     # output folder
    os.makedirs(target_root, exist_ok=True)

    # Loop through all images in all subfolders
    for ext in exts:
        for img_path in glob(os.path.join(source_root, "**", ext), recursive=True):
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            # Keep the same relative path structure
            rel_path = os.path.relpath(img_path, source_root)
            save_path = os.path.join(target_root, rel_path)

            # Make sure target subfolder exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save processed image
            cv2.imwrite(save_path, resized)

    print(f"Processing complete. Images saved in: {target_root}")


def _to_rgb(img):
    # cv2.imread returns BGR; keep it BGR internally and only convert when needed
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def random_resized_crop(img, out_size, scale=SCALE_RANGE, ratio=RATIO_RANGE):
    h, w = img.shape[:2]
    area = h * w

    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(*ratio)

        new_w = int(round(math.sqrt(target_area * aspect)))
        new_h = int(round(math.sqrt(target_area / aspect)))

        if new_w <= w and new_h <= h and new_w > 0 and new_h > 0:
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            crop = img[top:top+new_h, left:left+new_w]
            return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # Fallback: center crop square
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    crop = img[top:top+s, left:left+s]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def random_hflip(img, p=HFLIP_PROB):
    return cv2.flip(img, 1) if random.random() < p else img


def random_rotate(img, max_deg=ROT_MAX_DEG):
    angle = random.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    # Use reflect padding to avoid black corners
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def jitter_brightness_contrast(img, brightness=BRIGHTNESS, contrast=CONTRAST):
    # Contrast: alpha in [1-contrast, 1+contrast]
    alpha = 1.0 + random.uniform(-contrast, contrast)
    # Brightness: beta in [-brightness*255, +brightness*255]
    beta = random.uniform(-brightness, brightness) * 255.0
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def jitter_hue_saturation(img, saturation=SATURATION, hue=HUE):
    # img is BGR
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    # Hue shift (OpenCV H in [0,180])
    h_shift = random.uniform(-hue, hue) * 180.0
    h = (h + h_shift) % 180.0
    # Saturation scale
    s_scale = 1.0 + random.uniform(-saturation, saturation)
    s = np.clip(s * s_scale, 0, 255)
    hsv = cv2.merge([h, s, v])
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr


def maybe_grayscale(img, p=GRAYSCALE_PROB):
    if random.random() < p:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return img


def maybe_blur(img, p=BLUR_PROB):
    if random.random() < p:
        k = random.choice([3, 5])  # small, odd kernels only
        return cv2.GaussianBlur(img, (k, k), 0)
    return img


def augment_once(img_bgr):
    # 1) Geometry: random resized crop to IMG_SIZE
    x = random_resized_crop(img_bgr, IMG_SIZE, SCALE_RANGE, RATIO_RANGE)
    # After crop and resize, optional small rotation + flip
    x = random_rotate(x, ROT_MAX_DEG)
    x = random_hflip(x, HFLIP_PROB)
    # Ensure final size (rotate keeps same size, already square)
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 2) Photometric: brightness/contrast, hue/saturation, grayscale?, blur?
    x = jitter_brightness_contrast(x, BRIGHTNESS, CONTRAST)
    x = jitter_hue_saturation(x, SATURATION, HUE)
    x = maybe_grayscale(x, GRAYSCALE_PROB)
    x = maybe_blur(x, BLUR_PROB)

    return x


def save_augmented(img_path, img_aug, idx):
    base, ext = os.path.splitext(img_path)
    # Normalize extension to original ext
    out_path = f"{base}_aug{idx}{ext}"
    # Ensure directory exists
    cv2.imwrite(out_path, img_aug)


def process_folder(root=ROOT, n_aug=N_AUG_PER_IMAGE):
    # Find images
    patterns = [f"**/*{e}" for e in exts]
    all_imgs = []
    for p in patterns:
        all_imgs.extend(glob(os.path.join(root, p), recursive=True))

    print(f"Found {len(all_imgs)} images under '{root}'")

    count = 0
    for img_path in all_imgs:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = _to_rgb(img)
            if img is None:
                continue

            for i in range(1, n_aug + 1):
                aug = augment_once(img)
                save_augmented(img_path, aug, i)
            count += 1

            if count % 100 == 0:
                print(f"Processed {count} images...")
        except Exception as e:
            print(f"Error on {img_path}: {e}")

    print(f"Done. Augmented {count} images with {n_aug} variants each.")


if __name__ == "__main__":
    SPLIT_ROOT = "split_images"
    SOURCE_ROOT = "data_pairs_2"
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2  # TEST = 1 - TRAIN - VAL
    RANDOM_SEED = 42

    print("==> Splitting dataset")
    ensure_dir(SPLIT_ROOT)
    split_by_class(SOURCE_ROOT, SPLIT_ROOT, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)

    preprocessing()
    process_folder(ROOT, N_AUG_PER_IMAGE)