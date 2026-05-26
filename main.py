import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

def calculate_area(box):
    """Bounding Box [x1, y1, x2, y2]의 면적을 계산합니다."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def get_center(box):
    """Bounding Box [x1, y1, x2, y2]의 중심 좌표를 반환합니다."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def main(weights_path, image_path, conf_threshold=0.25):
    if not os.path.exists(image_path):
        print(f"오류: 이미지를 찾을 수 없습니다: {image_path}")
        return

    # 1. 모델 로드
    print(f"[{weights_path}] 모델을 불러옵니다...")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"오류: 모델 로드 실패: {e}")
        return

    # 2. 이미지 로드 및 추론
    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: 이미지 파일을 읽을 수 없습니다: {image_path}")
        return
        
    results = model(img, conf=conf_threshold)
    
    # 모델에서 클래스 이름 동적 로드
    # 0: box, 1: normal_hole, 2: crushed, 3: tear, 4: opened, 5: contamination (예상)
    model_names = model.names
    
    # 파손으로 간주할 클래스 ID (0번: box, 1번: normal_hole 제외)
    # IDs 2, 3, 4, 5 등이 파손에 해당함
    DAMAGE_CLASSES = [k for k, v in model_names.items() if k >= 2]
    BOX_CLASS_ID = 0
    NORMAL_HOLE_ID = 1

    # 재포장 판별 임계값 (%)
    REPACKAGE_THRESHOLD_PERCENT = 5.0

    boxes_info = []    # 전체 박스(box) 정보
    damages_info = []  # 파손 정보
    normal_holes_info = [] # 정상 구멍 정보 (시각화용)

    # 3. 추론 결과 파싱
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

    # 4. 객체들을 각 박스에 할당 (중심점 기준)
    def assign_to_box(objects, list_key):
        unassigned_count = 0
        for obj in objects:
            cx, cy = get_center(obj['bbox'])
            assigned = False
            for b_info in boxes_info:
                bx1, by1, bx2, by2 = b_info['bbox']
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    b_info[list_key].append(obj)
                    assigned = True
                    break
            if not assigned:
                unassigned_count += 1
        return unassigned_count

    unassigned_d = assign_to_box(damages_info, 'damages')
    unassigned_n = assign_to_box(normal_holes_info, 'normal_holes')

    if unassigned_d > 0:
        print(f"안내: {unassigned_d}개의 파손 부위가 박스 외부에서 검출되었습니다.")

    # 5. 시각화 및 결과 출력
    if not boxes_info:
        print("검출된 박스가 없습니다.")
        cv2.imwrite("result_output.jpg", img)
        return

    for i, b_info in enumerate(boxes_info):
        bx1, by1, bx2, by2 = map(int, b_info['bbox'])
        box_area = calculate_area(b_info['bbox'])
        
        # 파손 점수 계산
        total_damage_area = sum(calculate_area(d['bbox']) for d in b_info['damages'])
        damage_score = (total_damage_area / box_area) * 100 if box_area > 0 else 0
        needs_repackaging = damage_score >= REPACKAGE_THRESHOLD_PERCENT
        
        # [시각화] 박스 본체
        color = (0, 0, 255) if needs_repackaging else (0, 255, 0)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 3)

        status_text = "Repackage: YES" if needs_repackaging else "Repackage: NO"
        label_text = f"Box {i+1} | Score: {damage_score:.1f}% | {status_text}"
        
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (bx1, by1 - 25), (bx1 + tw, by1), color, -1)
        cv2.putText(img, label_text, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # [시각화] 파손 부위 (노란색)
        for damage in b_info['damages']:
            dx1, dy1, dx2, dy2 = map(int, damage['bbox'])
            class_name = model_names.get(damage['class_id'], f"ID:{damage['class_id']}")
            cv2.rectangle(img, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)
            cv2.putText(img, class_name, (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        # [시각화] 정상 구멍 (하늘색 - 점수에 포함 안됨)
        for nh in b_info['normal_holes']:
            nx1, ny1, nx2, ny2 = map(int, nh['bbox'])
            cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (255, 255, 0), 1)
            cv2.putText(img, "normal_hole", (nx1, ny1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    output_filename = "result_output.jpg"
    cv2.imwrite(output_filename, img)
    print(f"\n결과 이미지가 '{output_filename}'로 저장되었습니다. (박스: {len(boxes_info)}개)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="택배 파손 추론 및 재포장 판별 스크립트")
    parser.add_argument('--weights', type=str, required=True, help='학습된 모델 경로')
    parser.add_argument('--img', type=str, required=True, help='테스트할 이미지 경로')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값 (기본값: 0.25)')
    args = parser.parse_args()

    main(args.weights, args.img, args.conf)
