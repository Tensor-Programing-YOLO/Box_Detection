import os
from pathlib import Path
from collections import Counter
import yaml

def get_class_counts(label_dir):
    counts = Counter()
    label_files = list(Path(label_dir).glob("*.txt"))
    for label_file in label_files:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    counts[class_id] += 1
    return counts, len(label_files)

def main():
    with open("data.yaml", "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get("names", {})
    
    train_counts, train_files = get_class_counts("dataset/labels/train")
    val_counts, val_files = get_class_counts("dataset/labels/val")
    
    all_classes = sorted(set(train_counts.keys()) | set(val_counts.keys()))
    
    print(f"{'Class ID':<10} {'Class Name':<15} {'Train Count':<15} {'Train %':<10} {'Val Count':<15} {'Val %':<10}")
    print("-" * 80)
    
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    
    for cid in all_classes:
        name = class_names.get(cid, "Unknown")
        t_count = train_counts[cid]
        v_count = val_counts[cid]
        
        t_perc = (t_count / total_train * 100) if total_train > 0 else 0
        v_perc = (v_count / total_val * 100) if total_val > 0 else 0
        
        print(f"{cid:<10} {name:<15} {t_count:<15} {t_perc:>7.2f}%    {v_count:<15} {v_perc:>7.2f}%")
    
    print("-" * 80)
    print(f"{'Total':<10} {'':<15} {total_train:<15} {'100.00%':>8}    {total_val:<15} {'100.00%':>8}")
    print(f"\nTotal Images - Train: {train_files}, Val: {val_files}")

if __name__ == "__main__":
    main()
