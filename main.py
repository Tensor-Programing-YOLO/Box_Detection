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

def main(weights_path, image_path, conf_threshold=0.25, repackage_threshold=5.0):
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
    model_names = model.names
    
    # 파손으로 간주할 클래스 ID
    DAMAGE_CLASSES = [k for k, v in model_names.items() if k >= 2]
    BOX_CLASS_ID = 0
    NORMAL_HOLE_ID = 1

    boxes_info = []    # 전체 박스(box) 정보
    damages_info = []  # 파손 정보
    normal_holes_info = [] # 정상 구멍 정보

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

    # 4. 객체들을 각 박스에 할당 (교집합 면적 기준)
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

    # 5. 시각화 및 결과 출력
    if not boxes_info:
        print("검출된 박스가 없습니다. 이미지를 저장하지 않습니다.")
        return

    # [추가] 실제 정답(Ground Truth) 라벨 그리기
    def draw_ground_truth(img, image_path, model_names):
        # 이미지 경로에서 라벨 경로 추정 (images -> labels, 확장자 .jpg -> .txt)
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        
        if os.path.exists(label_path):
            h, w, _ = img.shape
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.split()
                if len(parts) < 5: continue
                
                class_id = int(parts[0])
                # YOLO 형식: class_id, x_center, y_center, width, height (0~1 사이값)
                x_c, y_c, bw, bh = map(float, parts[1:])
                
                # 픽셀 좌표로 변환
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
                
                class_name = model_names.get(class_id, f"GT:{class_id}")
                
                # 파란색 점선 느낌의 실선으로 GT 표시
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(img, f"GT: {class_name}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
            print(f"안내: 정답(GT) 라벨을 '{label_path}'에서 불러와 표시했습니다.")
        else:
            print(f"안내: 정답 라벨 파일을 찾을 수 없어 GT 시각화를 건너뜁니다. (경로: {label_path})")

    # 예측 결과 그리기 전 GT 먼저 그리기 (레이어 순서)
    draw_ground_truth(img, image_path, model_names)

    for i, b_info in enumerate(boxes_info):
        bx1, by1, bx2, by2 = map(int, b_info['bbox'])
        box_area = calculate_area(b_info['bbox'])
        
        # 파손 점수 계산 (중복 영역 합집합 처리)
        if b_info['damages'] and box_area > 0:
            mask_w = int(bx2 - bx1)
            mask_h = int(by2 - by1)
            if mask_w > 0 and mask_h > 0:
                damage_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
                for damage in b_info['damages']:
                    dx1, dy1, dx2, dy2 = map(int, damage['bbox'])
                    mx1 = max(0, dx1 - bx1)
                    my1 = max(0, dy1 - by1)
                    mx2 = min(mask_w, dx2 - bx1)
                    my2 = min(mask_h, dy2 - by1)
                    if mx2 > mx1 and my2 > my1:
                        damage_mask[my1:my2, mx1:mx2] = 1
                total_damage_area = np.sum(damage_mask)
            else:
                total_damage_area = 0
        else:
            total_damage_area = 0

        damage_score = (total_damage_area / box_area) * 100 if box_area > 0 else 0
        needs_repackaging = damage_score >= repackage_threshold
        
        # [시각화] 박스 본체
        color = (0, 0, 255) if needs_repackaging else (0, 255, 0)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 3)

        status_text = "Repackage: YES" if needs_repackaging else "Repackage: NO"
        label_text = f"Box {i+1} | Score: {damage_score:.1f}% | {status_text}"
        
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (bx1, by1 - 25), (bx1 + tw, by1), color, -1)
        cv2.putText(img, label_text, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for damage in b_info['damages']:
            dx1, dy1, dx2, dy2 = map(int, damage['bbox'])
            class_name = model_names.get(damage['class_id'], f"ID:{damage['class_id']}")
            cv2.rectangle(img, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)
            cv2.putText(img, class_name, (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        for nh in b_info['normal_holes']:
            nx1, ny1, nx2, ny2 = map(int, nh['bbox'])
            cv2.rectangle(img, (nx1, ny1), (nx2, ny2), (255, 255, 0), 1)
            cv2.putText(img, "normal_hole", (nx1, ny1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    output_filename = f"result_{os.path.basename(image_path)}"
    cv2.imwrite(output_filename, img)
    print(f"\n결과 이미지가 '{output_filename}'로 저장되었습니다. (박스: {len(boxes_info)}개)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="택배 파손 추론 및 재포장 판별 스크립트")
    parser.add_argument('--weights', type=str, required=True, help='학습된 가중치 경로 (예: yolo11m.pt)')
    parser.add_argument('--img', type=str, required=True, help='테스트 이미지 경로')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값 (기본값: 0.25)')
    parser.add_argument('--threshold', type=float, default=5.0, help='재포장 판별 면적 비율 임계값 (기본값: 5.0)')
    args = parser.parse_args()

    main(args.weights, args.img, args.conf, args.threshold)
