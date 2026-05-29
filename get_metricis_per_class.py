from ultralytics import YOLO
import os
import argparse

def main(weights_path):
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    # 모델 로드
    model = YOLO(weights_path)

    # 검증 수행
    print(f"Validating model: {weights_path}")
    results = model.val(data='data.yaml', split='val', verbose=False)

    # 클래스 이름 로드
    names = model.names

    print("\n" + "="*50)
    print(f"{'Class':<20} {'Images':<8} {'Instances':<10} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50-95':<10}")
    print("-" * 80)

    # 전체 결과 (mean)
    mp = results.results_dict['metrics/precision(B)']
    mr = results.results_dict['metrics/recall(B)']
    map50 = results.results_dict['metrics/mAP50(B)']
    map95 = results.results_dict['metrics/mAP50-95(B)']
    total_instances = results.nt_per_class.sum()
    print(f"{'all':<20} {'-':<8} {total_instances:<10.0f} {mp:<10.3f} {mr:<10.3f} {map50:<10.3f} {map95:<10.3f}")

    # 클래스별 결과
    for i, class_idx in enumerate(results.ap_class_index):
        class_name = names[class_idx]
        nt = results.nt_per_class[class_idx]
        p = results.box.p[i]
        r = results.box.r[i]
        ap50 = results.box.ap50[i]
        ap = results.box.ap[i] # mAP50-95
        
        print(f"{class_name:<20} {'-':<8} {nt:<10.0f} {p:<10.3f} {r:<10.3f} {ap50:<10.3f} {ap:<10.3f}")
    
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="클래스별 모델 성능 평가 스크립트")
    parser.add_argument('--weights', type=str, default='runs/detect/box_damage_640_tuned/weights/best.pt', help='학습된 가중치 파일 경로')
    args = parser.parse_args()
    
    main(args.weights)
