import os
import torch
from ultralytics import YOLO
from pathlib import Path

# ==============================================================================    
# [1] USER CONFIGURATION (사용자 설정 변수)
# ==============================================================================
selected_weight = 'yolo11m.pt'     # 베이스 모델 가중치 (예: yolo11n.pt, yolo11m.pt)
epochs = 300                         # 학습 반복 횟수
batch_size = 16                   # 배치 사이즈
img_size = 640                  # 입력 이미지 크기
custom_tag = 'mixed_v2_jaeunn+_conf'  # 실험 고유 태그 (폴더명에 포함됨)
# ==============================================================================

def main():
    # 실험 폴더명 동적 생성 규칙: {model}_{epochs}epoch_{tag}
    model_stem = Path(selected_weight).stem
    exp_name = f"{model_stem}_{epochs}epoch_{custom_tag}"
    
    # [수정] 결과를 하나의 폴더로 통합하기 위해 project와 name 설정 변경
    project_dir = 'results' 
    exp_path = exp_name
    
    # 디바이스 설정 (CUDA 사용 가능 시 GPU, 아니면 CPU)
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"\n>>> [실험 개시] {exp_name} | Device: {device}")

    # 1. 모델 로드 및 학습
    model = YOLO(selected_weight)
    
    results = model.train(
        data='data.yaml',
        epochs=epochs,
        patience=50,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        # seed=42,             
        deterministic=True,
        project=project_dir,
        name=exp_path,
        exist_ok=True, 
        
        # # --- [기존 적용: 조도 조절] ---
        # hsv_h=0.015,
        # hsv_s=0.7,
        # hsv_v=0.6,

        # Threshold 조절
        conf=0.15
        
        # # # --- [기하학 왜곡] ---
        # perspective=0.001, # 카메라 시점
        # shear=3.0, # 비틀림
        # degrees=5.0, # 회전각도
        # scale=0.2, # 상자의 높이 편차
        
        # # # --- [수정 1: 합성 Augmentation 조정] ---
        # mixup=0.15,
        # copy_paste=0.1,
        # erasing=0.0,    # 🚀 치명적 에러 방지: 꼬마 박스가 지워지는 것을 막기 위해 비활성화

        # # # --- [손실 함수(채점 기준) 극단적 미세 조정] ---
        # # cls=3.0,             # "종류(crushed/tear) 헷갈리면 벌점 3배!"
        # # box=7.5,             # "박스 위치 빗나가면 기본 벌점!"
        # # dfl=2.0              # 🚀 [수정 2] "테두리 1픽셀이라도 어긋나면 벌점 2배! 정밀하게 쳐라!"
        )

    # 2. 최적 가중치(best.pt) 자동 로드 및 검증 수행
    # model.train()의 결과 객체나 model.trainer에서 실제 저장 경로를 가져옵니다.
    save_dir = Path(model.trainer.save_dir)
    best_weights_path = save_dir / 'weights' / 'best.pt'
        
    if not best_weights_path.exists():
        print(f"Error: 최적 가중치 파일을 찾을 수 없습니다. ({best_weights_path})")
        return

    print(f"\n>>> [자동 평가] 최적 가중치로 상세 성능 지표를 추출합니다...")
    best_model = YOLO(best_weights_path)
    # [수정] 평가 결과도 동일한 폴더에 저장되도록 project, name 설정 추가
    val_results = best_model.val(
        data='data.yaml', 
        split='val', 
        verbose=False,
        project=project_dir,
        name=exp_path,
        # exist_ok=True # 돌일폴더면 지우기
    )

    # 3. 클래스별 상세 성적표 데이터 추출 및 포맷팅
    names = best_model.names
    report_lines = []
    
    header = f"{'Class':<20} {'Images':<8} {'Instances':<10} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50-95':<10}"
    separator = "=" * 85
    sub_sep = "-" * 85
    
    report_lines.append("=" * 30)
    report_lines.append("  FINAL METRICS REPORT")
    report_lines.append("=" * 30)
    report_lines.append(f"Experiment: {exp_name}")
    report_lines.append(f"Best Weights Path: {os.path.abspath(best_weights_path)}")
    report_lines.append("\n" + separator)
    report_lines.append(header)
    report_lines.append(separator)

    # 전체 평균 지표 (mean metrics)
    mp = val_results.results_dict['metrics/precision(B)']
    mr = val_results.results_dict['metrics/recall(B)']
    map50 = val_results.results_dict['metrics/mAP50(B)']
    map95 = val_results.results_dict['metrics/mAP50-95(B)']
    total_instances = val_results.nt_per_class.sum()
    
    all_row = f"{'all (mean)':<20} {'-':<8} {total_instances:<10.0f} {mp:<10.3f} {mr:<10.3f} {map50:<10.3f} {map95:<10.3f}"
    report_lines.append(all_row)
    report_lines.append(sub_sep)

    # 개별 클래스별 지표 추출
    for i, class_idx in enumerate(val_results.ap_class_index):
        class_name = names[class_idx]
        nt = val_results.nt_per_class[class_idx]
        p = val_results.box.p[i]
        r = val_results.box.r[i]
        ap50 = val_results.box.ap50[i]
        ap = val_results.box.ap[i] # mAP50-95
        
        class_row = f"{class_name:<20} {'-':<8} {nt:<10.0f} {p:<10.3f} {r:<10.3f} {ap50:<10.3f} {ap:<10.3f}"
        report_lines.append(class_row)
    
    report_lines.append(separator)

    # [추가] 사용자가 요청한 주요 지표 (미사용 지표 포함 여부 확인용)
    # YOLO의 기본 train() 및 val() 함수가 다음 그래프들은 자동으로 생성하여 폴더에 저장합니다:
    # - Train/Val Box, Cls, DFL Loss: results.png
    # - Confusion Matrix: confusion_matrix.png
    # - PR Curve: BoxPR_curve.png
    # 위 항목들은 코드 상에서 명시적으로 추출하지 않아도 'results' 폴더 내에 생성됩니다.

    # 4. 터미널 출력 및 텍스트 파일 저장
    final_report = "\n".join(report_lines)
    print("\n" + final_report + "\n")

    # [수정] 보고서를 결과 폴더 바로 아래에 저장
    report_save_path = best_weights_path.parent.parent / 'final_metrics_report.txt'
    with open(report_save_path, 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    print(f">>> [완료] 학습 로그, 시각화 그래프 및 상세 성적표가 저장되었습니다.")
    print(f">>> 저장 경로: {os.path.abspath(report_save_path.parent)}")

if __name__ == '__main__':
    main()