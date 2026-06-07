import os
import glob
import argparse
from main import main as inference_main

#하단 복붙해서 실행하시면 돌아갑니다.
#python test.py --weights runs/detect/results/yolov8m_300epoch_augmented_hypertuned/weights/best.pt --dir dataset/images/val --conf 0.1

def batch_test(weights_path, img_dir, conf, threshold):
    # 지원하는 이미지 확장자
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in exts:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    
    if not img_files:
        print(f"[{img_dir}]에서 이미지를 찾을 수 없습니다.")
        return

    print(f"총 {len(img_files)}개의 이미지에 대해 테스트를 시작합니다.")
    
    # 결과 저장 폴더 생성
    output_dir = "DAMAGE_SCORE_test_results"
    os.makedirs(output_dir, exist_ok=True)

    for img_path in img_files:
        print(f"\n처리 중: {img_path}")
        # main.py의 main 함수 호출 (출력 경로 로직은 main.py 내부에 있음)
        # 단, main.py에서 output_filename을 현재 폴더에 생성하므로 
        # 필요시 main.py를 수정하여 경로를 지정할 수 있게 할 수 있습니다.
        inference_main(weights_path, img_path, conf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="배치 이미지 테스트 스크립트")
    parser.add_argument('--weights', type=str, required=True, help='학습된 가중치 경로')
    parser.add_argument('--dir', type=str, default='dataset/images/val', help='테스트할 이미지 디렉토리')
    parser.add_argument('--conf', type=float, default=0.05, help='신뢰도 임계값')
    parser.add_argument('--threshold', type=float, default=5.0, help='재포장 판별 임계값')
    args = parser.parse_args()

    batch_test(args.weights, args.dir, args.conf, args.threshold)
