import os
import shutil
import random
from pathlib import Path


def preprocess_dataset():
    # 1. 경로 설정
    raw_images_dir = Path("raw_dataset/images")
    raw_labels_dir = Path("raw_dataset/labels")
    
    target_base = Path("dataset")
    dirs = {
        "train_img": target_base / "images/train",
        "val_img": target_base / "images/val",
        "train_lbl": target_base / "labels/train",
        "val_lbl": target_base / "labels/val",
    }

    # 타겟 디렉토리 생성
    for d in dirs.values():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 2. 수동 경계 설정 (새로운 상자가 시작되는 사진 번호)
    # 이 번호들을 기준으로 사진들이 그룹화되어 Train/Val로 나뉩니다.
    boundaries = [
        1, 109, 204, 309, 410, 509, 609, 713, 808, 900, 1001, 1101, 
        1201, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 
        2200, 2300, 2400, 2503, 2600, 2701, 2800, 2901, 3000, 3101, 
        3201, 3222, 3244, 3267, 3293, 3321, 3349, 3377, 3381, 3383, 3386, 3389, 3391, 3393, 3396, 3400, 3402,  3404, 3434, 
        3470, 3486, 3498, 3521, 3541, 3550, 3553, 3577, 3589, 3593 ,3596, 3603, 3604, 3605, 3608, 3609, 3612, 3613, 3614, 3617, 3624, 3630, 3637
    ]
    boundary_set = set(boundaries)

    # 3. 데이터 그룹화
    image_files = sorted([f for f in raw_images_dir.glob("*.jpg")])
    groups = {}
    current_group_id = 0

    for img_path in image_files:
        try:
            # 파일명에서 숫자 추출 (예: 00001.jpg -> 1)
            file_num = int(img_path.stem)
        except ValueError:
            continue

        if file_num in boundary_set:
            current_group_id += 1
        
        if current_group_id not in groups:
            groups[current_group_id] = []
        groups[current_group_id].append(img_path)

    # 4. 상자 그룹 단위 셔플 및 분할 (8:2)
    group_ids = list(groups.keys())
    random.seed(42) # 재현성을 위해 시드 고정
    random.shuffle(group_ids)

    split_idx = int(len(group_ids) * 0.8)
    train_groups = group_ids[:split_idx]
    val_groups = group_ids[split_idx:]

    def copy_files(selected_groups, img_target, lbl_target):
        count = 0
        discard_count = 0
        ignore_class_id = 5  # 'contamination' 클래스 ID
        
        for g_id in selected_groups:
            for img_path in groups[g_id]:
                lbl_path = raw_labels_dir / f"{img_path.stem}.txt"
                
                # contamination 클래스 포함 여부 확인
                has_contamination = False
                if lbl_path.exists():
                    with open(lbl_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and int(parts[0]) == ignore_class_id:
                                has_contamination = True
                                break
                
                # contamination이 있으면 해당 이미지/라벨 세트 전체를 버림
                if has_contamination:
                    discard_count += 1
                    continue

                # 이미지 복사
                shutil.copy(img_path, img_target / img_path.name)
                
                # 라벨 복사
                if lbl_path.exists():
                    shutil.copy(lbl_path, lbl_target / lbl_path.name)
                
                count += 1
        return count, discard_count

    # 5. 파일 복사 실행
    train_count, train_discard = copy_files(train_groups, dirs["train_img"], dirs["train_lbl"])
    val_count, val_discard = copy_files(val_groups, dirs["val_img"], dirs["val_lbl"])

    # 결과 출력
    print(f"총 상자 그룹 수: {len(group_ids)}")
    print(f"Train 이미지 수: {train_count} (버려진 사진: {train_discard})")
    print(f"Val 이미지 수: {val_count} (버려진 사진: {val_discard})")
    print(f"총 제외된 contamination 사진 수: {train_discard + val_discard}")

if __name__ == "__main__":
    preprocess_dataset()
