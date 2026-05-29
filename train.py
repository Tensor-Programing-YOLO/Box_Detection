import torch
from ultralytics import YOLO

def main():
    # ==========================================
    # [사용자 설정] 모델 및 하이퍼파라미터
    # ==========================================
    # 사용할 모델을 아래에 지정하세요. 
    # 사용 가능 모델 예시: 'yolov8n.pt', 'yolov10n.pt', 'yolo11n.pt'
    selected_weight = 'yolo11m.pt'
    
    epochs = 150      # 학습 에포크 수
    batch = 8         # 배치 사이즈
    img_size = 640    # 입력 이미지 크기
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"[{selected_weight}] 모델을 불러옵니다... (Device: {device})")

    # 모델 로드
    model = YOLO(selected_weight)

    # 학습 시작
    print("학습을 시작합니다...")
    results = model.train(
        data='data.yaml',
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        
        # --- [추가] 희소 클래스(tear, crushed) 구출용 파라미터 ---
        cos_lr=True,
        
        # --- [수정] 픽셀 대비를 극대화하여 찢어짐(tear)을 강조 ---
        hsv_h=0.015, 
        hsv_s=0.9,
        hsv_v=0.7,
        
        mosaic=0.0, mixup=0.0, scale=0.1, degrees=10.0,
        
        project='runs/detect',
        name='box_damage_640_tuned',
        device=device
    )
    
    print("학습이 완료되었습니다!")

if __name__ == '__main__':
    main()
