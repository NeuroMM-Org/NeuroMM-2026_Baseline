import os
import pandas as pd


DEFAULT_MANIFEST_PATH = "neuromm26_datasets/annotations/neuromm2026_train_val_patient_split.csv"


def check_missing_features(manifest_path: str = DEFAULT_MANIFEST_PATH):
    feature_dir = "neuromm26_datasets/processed/features/video/clip-base"

    print(f"加载 Manifest 文件: {manifest_path}")
    df = pd.read_csv(manifest_path)
    total_rows = len(df)
    print(f"CSV 总行数: {total_rows}")

    missing_path_df = df[df["raw_video_relpath"].isna()]
    if len(missing_path_df) > 0:
        print(f"\n[警告] 发现 {len(missing_path_df)} 个样本在 CSV 中缺少 'raw_video_relpath' (导致被静默跳过):")
        print(missing_path_df["sample_id"].tolist())

    duplicate_ids = df[df.duplicated(subset=["sample_id"], keep=False)]
    if len(duplicate_ids) > 0:
        print(f"\n[警告] 发现 {len(duplicate_ids)} 条记录具有重复的 'sample_id' (会导致 .npy 被覆盖):")
        print(duplicate_ids["sample_id"].unique().tolist())

    expected_ids = set(df["sample_id"].astype(str))

    if not os.path.exists(feature_dir):
        print(f"\n[错误] 特征目录不存在: {feature_dir}")
        return

    actual_files = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
    actual_ids = set(f.replace(".npy", "") for f in actual_files)

    print(f"\n期望的唯一 sample_id 数量: {len(expected_ids)}")
    print(f"实际生成的特征文件数量: {len(actual_ids)}")

    missing_ids = expected_ids - actual_ids
    print(f"\n---> 共有 {len(missing_ids)} 个样本的视频特征提取失败或缺失！<---")

    if missing_ids:
        print("\n详细缺失名单及对应视频路径：")
        missing_df = df[df["sample_id"].astype(str).isin(missing_ids)]
        for _, row in missing_df.iterrows():
            sid = row["sample_id"]
            vpath = row.get("raw_video_relpath", "路径为空(NaN)")
            print(f"Sample ID: {sid: <15} | Video Path: {vpath}")


if __name__ == "__main__":
    check_missing_features()
