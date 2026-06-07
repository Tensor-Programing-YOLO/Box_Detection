import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

# 클래스별 위험도 가중치 (ISTA 기반)
CLASS_WEIGHTS = {
    0: 0,    # box
    1: 0,    # normal_hole
    2: 50,   # crushed
    3: 90,   # tear
    4: 100,  # opened
}

def calculate_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def main(weights_path, image_path, conf_threshold=0.25):
    if not os.path.exists(image_path):
        print(f"오류: 이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"[{weights_path}] 모델을 불러옵니다...")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"오류: 모델 로드 실패: {e}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: 이미지 파일을 읽을 수 없습니다: {image_path}")
        return

    results = model(img, conf=conf_threshold)
    model_names = model.names

    DAMAGE_CLASSES = [k for k, v in model_names.items() if k >= 2]
    BOX_CLASS_ID = 0
    NORMAL_HOLE_ID = 1

    boxes_info = []
    damages_info = []
    normal_holes_info = []

    for result in results:
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_id = int(class_id)
            bbox = [x1, y1, x2, y2]

            if class_id == BOX_CLASS_ID:
                boxes_info.append({
                    'bbox': bbox,
                    'score': score,
                    'damages': [],
                    'normal_holes': []
                })
            elif class_id in DAMAGE_CLASSES:
                damages_info.append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'score': score
                })
            elif class_id == NORMAL_HOLE_ID:
                normal_holes_info.append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'score': score
                })

    def assign_to_box(objects, list_key):
        unassigned_count = 0
        for obj in objects:
            best_box = None
            max_inter_area = 0
            for b_info in boxes_info:
                bx1, by1, bx2, by2 = b_info['bbox']
                ox1, oy1, ox2, oy2 = obj['bbox']
                x_left = max(bx1, ox1)
                y_top = max(by1, oy1)
                x_right = min(bx2, ox2)
                y_bottom = min(by2, oy2)
                if x_right > x_left and y_bottom > y_top:
                    inter_area = (x_right - x_left) * (y_bottom - y_top)
                    if inter_area > max_inter_area:
                        max_inter_area = inter_area
                        best_box = b_info
            if best_box is not None and max_inter_area > 0:
                best_box[list_key].append(obj)
            else:
                unassigned_count += 1
        return unassigned_count

    unassigned_d = assign_to_box(damages_info, 'damages')
    unassigned_n = assign_to_box(normal_holes_info, 'normal_holes')

    if unassigned_d > 0:
        print(f"안내: {unassigned_d}개의 파손 부위가 박스 외부에서 검출되었습니다.")

    if not boxes_info:
        print("검출된 박스가 없습니다.")
        return

    def draw_ground_truth(img, image_path, model_names):
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        if os.path.exists(label_path):
            h, w, _ = img.shape
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                x_c, y_c, bw, bh = map(float, parts[1:])
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
                class_name = model_names.get(class_id, f"GT:{class_id}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(img, f"GT: {class_name}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

    draw_ground_truth(img, image_path, model_names)

    for i, b_info in enumerate(boxes_info):
        bx1, by1, bx2, by2 = map(int, b_info['bbox'])
        box_area = calculate_area(b_info['bbox'])

        # ── ISTA 기반 damage score 계산 ──────────────────────
        total_score = 0
        if box_area > 0:
            for damage in b_info['damages']:
                cls_id = damage['class_id']
                confidence = damage['score']
                d_area = calculate_area(damage['bbox'])
                bbox_ratio = d_area / box_area
                weight = CLASS_WEIGHTS.get(cls_id, 30)
                total_score += weight * confidence + bbox_ratio * 100

        damage_score = min(total_score, 100)

        # ── PASS / WARNING / REJECT 판정 ─────────────────────
        if damage_score <= 20:
            status = "PASS"
            color = (0, 200, 0)
        elif damage_score <= 60:
            status = "WARNING"
            color = (0, 165, 255)
        else:
            status = "REJECT"
            color = (0, 0, 255)

        # ── 박스 그리기 ───────────────────────────────────────
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 3)

        label_text = f"Box {i+1} | Score: {damage_score:.1f} | {status}"
        font_scale = max(0.4, min(1.2, img.shape[1] / 1000))
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (bx1, by1), (bx1 + tw, by1 + th + 10), color, -1)
        cv2.putText(img, label_text, (bx1, by1 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # ── 결함 바운딩박스 ───────────────────────────────────
        for damage in b_info['damages']:
            dx1, dy1, dx2, dy2 = map(int, damage['bbox'])
            class_name = model_names.get(damage['class_id'], f"ID:{damage['class_id']}")
            cv2.rectangle(img, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
            cv2.putText(img, class_name, (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ── 정상 구멍 바운딩박스 ──────────────────────────────
        for nh in b_info['normal_holes']:
            nx1, ny1, nx2, ny2 = map(int, nh['bbox'])
            cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (255, 255, 0), 1)
            cv2.putText(img, "normal_hole", (nx1, ny1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    output_dir = "DAMAGE_SCORE_test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
    cv2.imwrite(output_filename, img)
    print(f"\n결과 이미지 저장 완료: '{output_filename}' (박스: {len(boxes_info)}개)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="택배 파손 추론 스크립트")
    parser.add_argument('--weights', type=str, required=True, help='학습된 가중치 경로')
    parser.add_argument('--img', type=str, required=True, help='테스트 이미지 경로')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값 (기본값: 0.25)')
    args = parser.parse_args()

    main(args.weights, args.img, args.conf)