"""
augment.py
───────────
실행 위치: Box_Detection/
전처리(preprocess.py) 실행 후 돌릴 것!

crushed(class 2), tear(class 3) 를 dataset/images/train 에 바로 증강 추가.
원본 파일명과 구분되게 _aug 접미사 붙임.

설치: pip install albumentations opencv-python
"""

import cv2
import random
import numpy as np
from pathlib import Path
import albumentations as A

# ── 설정 ────────────────────────────────────────────────
TRAIN_IMG_DIR = Path("dataset/images/train")
TRAIN_LBL_DIR = Path("dataset/labels/train")

CLASS_NAMES = ["box", "normal_hole", "crushed", "tear", "opened"]

# 증강할 클래스 → 목표 장수
TARGETS = {
    2: 300,  # crushed
    3: 200,  # tear
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5,
                       border_mode=cv2.BORDER_CONSTANT),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


def load_labels(lbl_path):
    boxes, classes = [], []
    for line in lbl_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        classes.append(int(float(parts[0])))
        boxes.append([float(x) for x in parts[1:5]])
    return classes, boxes


def save_labels(lbl_path, classes, boxes):
    lines = [f"{c} {' '.join(f'{v:.6f}' for v in b)}" for c, b in zip(classes, boxes)]
    lbl_path.write_text("\n".join(lines))


def get_sources(cls_idx):
    result = []
    for lbl in TRAIN_LBL_DIR.glob("*.txt"):
        classes, _ = load_labels(lbl)
        if cls_idx in classes:
            for ext in [".jpg", ".png", ".jpeg"]:
                img = TRAIN_IMG_DIR / (lbl.stem + ext)
                if img.exists():
                    result.append((img, lbl))
                    break
    return result


def count_class(cls_idx):
    count = 0
    for lbl in TRAIN_LBL_DIR.glob("*.txt"):
        classes, _ = load_labels(lbl)
        count += classes.count(cls_idx)
    return count


def augment_class(cls_idx, target):
    name = CLASS_NAMES[cls_idx]
    current = count_class(cls_idx)
    needed = target - current

    if needed <= 0:
        print(f"  [{name}] 이미 {current}장 → 스킵")
        return

    print(f"  [{name}] {current}장 → {target}장 목표 ({needed}장 생성)")

    sources = get_sources(cls_idx)
    if not sources:
        print(f"  [{name}] 소스 없음 → 스킵")
        return

    generated = 0
    attempts = 0

    while generated < needed and attempts < needed * 10:
        attempts += 1
        img_path, lbl_path = random.choice(sources)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        classes, boxes = load_labels(lbl_path)
        if not boxes:
            continue
        try:
            result = transform(image=img_rgb, bboxes=boxes, class_labels=classes)
        except Exception:
            continue
        if not result['bboxes']:
            continue

        new_stem = f"{img_path.stem}_aug{cls_idx}_{generated:04d}"
        new_img = TRAIN_IMG_DIR / (new_stem + img_path.suffix)
        new_lbl = TRAIN_LBL_DIR / (new_stem + ".txt")

        cv2.imwrite(str(new_img), cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR))
        save_labels(new_lbl, result['class_labels'], result['bboxes'])
        generated += 1

    print(f"  [{name}] {generated}장 생성 완료 ✅")


def main():
    print("🔧 증강 시작!\n")
    for cls_idx, target in TARGETS.items():
        augment_class(cls_idx, target)
    print("\n✅ 증강 완료! → python train.py 실행해봐")


if __name__ == "__main__":
    main()
