from ultralytics import YOLO

def main():
    # ==========================================
    # [사용자 설정] 모델 및 하이퍼파라미터
    # ==========================================
    # 사용할 모델을 아래에 지정하세요. 
    # 사용 가능 모델 예시: 'yolov8n.pt', 'yolov10n.pt', 'yolo11n.pt'
    selected_weight = 'yolov8n.pt'
    
    epochs = 50       # 학습 에포크 수
    batch = 16        # 배치 사이즈
    img_size = 640    # 입력 이미지 크기
    
    print(f"[{selected_weight}] 모델을 불러옵니다...")

    # 모델 로드
    model = YOLO(selected_weight)

    # 학습 시작
    print("학습을 시작합니다...")
    results = model.train(
        data='data.yaml',       # yaml 설정 파일 경로
        epochs=epochs,          # 에포크 수
        batch=batch,            # 배치 크기
        imgsz=img_size,         # 이미지 사이즈
        project='runs/detect',  # 결과 저장 디렉토리
        name=f'box_damage_{selected_weight.replace(".pt", "")}', # 실험 이름 (예: runs/detect/box_damage_yolov8n)
        device='auto'           # 'auto'로 설정하여 GPU가 있으면 사용하고, 없으면 CPU 사용
    )
    
    print("학습이 완료되었습니다!")

if __name__ == '__main__':
    main()
