import os
import glob
import random
import shutil
from pathlib import Path

# ==========================================
# [사용자 설정] 원본 데이터의 클래스 ID 매핑
# ==========================================
# 원본 클래스 ID : 최종 변경될 클래스 ID
CLASS_MAPPING = {
    0: 0,   # 원본 box -> 0 (box)
    1: 1,   # 원본 normal_hole -> 1 (normal_hole)
    2: 2,   # 원본 crushed -> 2 (crushed)
    3: 3,   # 원본 tear -> 3 (tear)
    5: 4,   # 원본 opened -> 4 (opened)
    7: 5    # 원본 contamination -> 5 (contamination)
}

def process_and_split(all_images_dir, all_labels_dir, output_base_dir, split_ratio=0.8):
    """
    all_labels/ 폴더의 모든 라벨을 읽고 정제한 뒤 무작위로 섞어
    train/val로 분할하고 해당 이미지와 함께 dataset/ 하위 폴더로 복사합니다.
    """
    all_labels_path = Path(all_labels_dir)
    all_images_path = Path(all_images_dir)
    output_path = Path(output_base_dir)

    # 1. 대상 폴더 생성 (dataset/images/train, val 및 dataset/labels/train, val)
    train_img_dir = output_path / 'images' / 'train'
    val_img_dir = output_path / 'images' / 'val'
    train_lbl_dir = output_path / 'labels' / 'train'
    val_lbl_dir = output_path / 'labels' / 'val'

    # 폴더가 없으면 자동으로 생성 (os.makedirs와 동일하게 작동)
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. 전체 텍스트 파일(라벨) 목록 수집
    txt_files = list(all_labels_path.glob('*.txt'))
    if not txt_files:
        print(f"[{all_labels_dir}] 폴더에 텍스트 파일이 없습니다.")
        return

    print(f"총 {len(txt_files)}개의 라벨 파일을 찾았습니다. 무작위 분할(8:2)을 시작합니다...")
    
    # 3. 무작위 섞기 (shuffle)
    # 재현성을 원하시면 random.seed(42) 등을 추가하세요.
    random.shuffle(txt_files)

    # 4. 8:2 비율 분할 계산
    split_index = int(len(txt_files) * split_ratio)
    train_files = txt_files[:split_index]
    val_files = txt_files[split_index:]

    # 이미지 파일 확장자 (원본 이미지가 어떤 확장자인지 모를 경우를 대비)
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    def process_subset(file_list, target_img_dir, target_lbl_dir, subset_name):
        processed_count = 0
        skipped_count = 0
        for txt_file in file_list:
            # ---------------------------------------------
            # [1] 라벨 파일 전처리(클래스 변경/삭제) 및 저장
            # ---------------------------------------------
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                class_id = int(parts[0])

                if class_id in CLASS_MAPPING:
                    new_class_id = CLASS_MAPPING[class_id]
                    parts[0] = str(new_class_id)
                    new_lines.append(' '.join(parts) + '\n')
                # 매핑에 없는 클래스(예: 병합, 삭제 대상)는 무시되어 저장되지 않음

            # [1-1] 박스(class 0)가 2개 이상인 경우 제외
            box_count = sum(1 for line in new_lines if line.split()[0] == '0')
            if box_count >= 2:
                skipped_count += 1
                continue

            # ---------------------------------------------
            # [2] 대응되는 이미지 파일 찾아서 복사
            # ---------------------------------------------
            base_name = txt_file.stem # 확장자(.txt)를 제외한 파일명
            img_found = False
            found_img_path = None
            
            # 대소문자 구분 없이 확장자 확인
            for ext in img_extensions:
                for actual_ext in [ext.lower(), ext.upper()]:
                    img_candidate = all_images_path / (base_name + actual_ext)
                    if img_candidate.exists():
                        found_img_path = img_candidate
                        img_found = True
                        break
                if img_found: break
            
            if img_found:
                # 이미지가 있을 때만 라벨 파일 저장
                new_lbl_path = target_lbl_dir / txt_file.name
                with open(new_lbl_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                
                shutil.copy(found_img_path, target_img_dir / found_img_path.name)
                processed_count += 1
            else:
                print(f"경고: {txt_file.name} 라벨에 대응하는 이미지를 찾을 수 없습니다. 건너뜁니다.")
            
        print(f"[{subset_name}] 완료: {processed_count}개 생성 (박스 과다로 {skipped_count}개 제외)")

    # 5. 분할된 리스트를 바탕으로 처리 진행
    process_subset(train_files, train_img_dir, train_lbl_dir, "Train (80%)")
    process_subset(val_files, val_img_dir, val_lbl_dir, "Validation (20%)")

if __name__ == '__main__':
    base_dir = Path(__file__).parent
    
    # ---------------------------------------------------------
    # 작업 전, 아래 경로에 원본 데이터가 모두 모여있어야 합니다.
    # ---------------------------------------------------------
    source_labels = base_dir / 'dataset' / 'labels' # 원본 txt 파일들이 모두 모여있는 폴더
    source_images = base_dir / 'dataset' / 'images' # 원본 이미지 파일들이 모두 모여있는 폴더
    output_dataset = base_dir / 'dataset'   # 최종 생성될 train/val 폴더의 최상위 경로

    print("전처리 및 데이터셋 분할(Train/Val) 파이프라인 시작...\n")
    
    if not source_labels.exists():
        print(f"오류: 라벨 폴더({source_labels})가 존재하지 않습니다.")
    elif not source_images.exists():
        print(f"오류: 이미지 폴더({source_images})가 존재하지 않습니다.")
    else:
        # 비율(split_ratio) 조절 시 0.8 부분을 변경하면 됩니다.
        process_and_split(source_images, source_labels, output_dataset, split_ratio=0.8)
        print("\n모든 라벨 정제 및 Train/Val 이미지-라벨 복사가 완료되었습니다!")