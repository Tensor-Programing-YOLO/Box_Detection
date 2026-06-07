import cv2
from pathlib import Path
import shutil

# =========================
# 경로 설정
# =========================
ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "raw_dataset" / "images"
LABEL_DIR = ROOT / "raw_dataset" / "labels"
OUTPUT_DIR = ROOT / "visualization_output"

# data.yaml에서 클래스명 읽기 시도
DATA_YAML = ROOT / "data.yaml"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def load_class_names():
    """
    data.yaml 안의 names를 읽어옵니다.
    없으면 클래스 id 그대로 표시합니다.
    """
    if not DATA_YAML.exists():
        return None

    try:
        import yaml

        with open(DATA_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data.get("names", None)

        if isinstance(names, list):
            return names

        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]

    except Exception as e:
        print(f"[경고] data.yaml을 읽지 못했습니다: {e}")

    return None


def yolo_to_xyxy(label, img_w, img_h):
    """
    YOLO 형식:
    class_id x_center y_center width height
    값을 실제 이미지 좌표 x1, y1, x2, y2로 변환
    """
    class_id, x_center, y_center, box_w, box_h = label

    x1 = int((x_center - box_w / 2) * img_w)
    y1 = int((y_center - box_h / 2) * img_h)
    x2 = int((x_center + box_w / 2) * img_w)
    y2 = int((y_center + box_h / 2) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return int(class_id), x1, y1, x2, y2


def draw_boxes(image_path, label_path, save_path, class_names=None):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[스킵] 이미지를 읽을 수 없음: {image_path}")
        return 0

    img_h, img_w = image.shape[:2]
    box_count = 0

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(f"[경고] 라벨 형식 이상: {label_path} / {line}")
                continue

            values = list(map(float, parts[:5]))
            class_id, x1, y1, x2, y2 = yolo_to_xyxy(values, img_w, img_h)

            # 클래스별 색상
            color = (
                int((class_id * 70 + 50) % 255),
                int((class_id * 130 + 80) % 255),
                int((class_id * 200 + 120) % 255),
            )

            # 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 라벨 텍스트
            if class_names and class_id < len(class_names):
                label_text = class_names[class_id]
            else:
                label_text = f"class_{class_id}"

            # 텍스트 배경
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_w, text_h = text_size

            text_bg_y1 = max(0, y1 - text_h - 8)
            text_bg_y2 = y1

            cv2.rectangle(
                image,
                (x1, text_bg_y1),
                (x1 + text_w + 8, text_bg_y2),
                color,
                -1,
            )

            cv2.putText(
                image,
                label_text,
                (x1 + 4, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            box_count += 1

    # 저장 폴더 생성
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)

    return box_count


def main():
    if not IMAGE_DIR.exists():
        print(f"[오류] 이미지 폴더가 없습니다: {IMAGE_DIR}")
        return

    if not LABEL_DIR.exists():
        print(f"[오류] 라벨 폴더가 없습니다: {LABEL_DIR}")
        return

    # 기존 결과 삭제 후 새로 생성
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names()

    image_files = []
    for ext in IMAGE_EXTS:
        image_files.extend(IMAGE_DIR.rglob(f"*{ext}"))
        image_files.extend(IMAGE_DIR.rglob(f"*{ext.upper()}"))

    image_files = sorted(image_files)

    if not image_files:
        print(f"[오류] 이미지 파일을 찾을 수 없습니다: {IMAGE_DIR}")
        return

    total_images = 0
    total_boxes = 0
    missing_labels = 0

    print(f"총 {len(image_files)}개 이미지 시각화 시작")

    for image_path in image_files:
        # images 하위 경로 구조 유지
        relative_path = image_path.relative_to(IMAGE_DIR)

        # 같은 상대경로의 labels txt 찾기
        label_path = LABEL_DIR / relative_path.with_suffix(".txt")

        # 결과 저장 경로
        save_path = OUTPUT_DIR / relative_path

        if not label_path.exists():
            missing_labels += 1

        box_count = draw_boxes(
            image_path=image_path,
            label_path=label_path,
            save_path=save_path,
            class_names=class_names,
        )

        total_images += 1
        total_boxes += box_count

        print(f"[완료] {relative_path} / boxes: {box_count}")

    print("\n========== 시각화 완료 ==========")
    print(f"처리한 이미지 수: {total_images}")
    print(f"그린 박스 수: {total_boxes}")
    print(f"라벨 없는 이미지 수: {missing_labels}")
    print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()