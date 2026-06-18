"""
train.py — Train Random Forest classifier
==========================================
Input  : data/processed/features_final.csv
Output : models/random_forest.pkl

Quy trình:
  1. Load features_final.csv
  2. Xử lý missing values cho các feature có thể = -1 (time_interval_std, video_upload_frequency, subscriber_velocity)
  3. Train/test split stratified
  4. Train Random Forest với class_weight='balanced' (xử lý imbalance 142:62)
  5. Cross-validation 5-fold
  6. Đánh giá: Accuracy, Precision, Recall, F1, ROC-AUC
  7. Feature importance
  8. Save model

Cách chạy:
    python src/train.py
    python src/train.py --no-save  # test không lưu model
"""

import sys
import logging
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
    RF_TEST_SIZE, RF_CV_FOLDS,
    Paths, LABEL_SLOP, LABEL_GENUINE,
    validate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature columns đưa vào model ────────────────────────────────────────────
# 15 features theo 4 nhóm:
#   Nhóm chuỗi thời gian & vận tốc đăng bài (5):
#     time_interval_std, upload_burst_ratio, video_upload_frequency,
#     if_anomaly_score, view_per_video
#   Nhóm dấu vết cấu trúc & định dạng văn bản AI (5):
#     dash_density, title_length_std, capitalization_ratio,
#     opening_repeat_ratio, temporal_clickbait_ratio
#   Nhóm độ đa dạng & tương đồng nội dung (2):
#     type_token_ratio, avg_title_similarity
#   Nhóm chỉ số tài chính & gian lận tương tác (3):
#     sub_to_view_ratio, subscriber_velocity, sub_to_view_velocity_ratio
FEATURE_COLS = [
    # Nhóm chuỗi thời gian & vận tốc đăng bài
    "time_interval_std",           # Khoảng cách đăng bài (std)
    "upload_burst_ratio",          # Tỷ lệ đăng dồn dập
    "video_upload_frequency",      # Tần suất đăng video (videos/ngày)
    "if_anomaly_score",            # Isolation Forest anomaly score
    "view_per_video",              # Lượt xem trung bình mỗi video

    # Nhóm dấu vết cấu trúc & định dạng văn bản AI
    "dash_density",                # Mật độ dấu gạch ngang
    "title_length_std",            # Std độ dài title
    "capitalization_ratio",        # Mật độ viết hoa
    "opening_repeat_ratio",        # Tỷ lệ lặp cụm mở đầu
    "temporal_clickbait_ratio",    # Cross-feature clickbait × time

    # Nhóm độ đa dạng & tương đồng nội dung
    "type_token_ratio",            # Độ đa dạng từ vựng
    "avg_title_similarity",        # Độ tương đồng title

    # Nhóm chỉ số tài chính & gian lận tương tác
    "sub_to_view_ratio",           # Sub / View
    "subscriber_velocity",         # Tốc độ tăng sub (subs/ngày)
    "sub_to_view_velocity_ratio",  # log10((sub_vel+1)/(view_vel+1))
]

# Danh sách các feature có thể có giá trị -1 (missing) và cần impute
MISSING_FEATURES = {
    "time_interval_std": "F1",
    "video_upload_frequency": "F14",
    "subscriber_velocity": "F17",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & prep data
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load features_final.csv, xử lý missing, trả về (df, X, y).
    """
    df = pd.read_csv(Paths.FEATURES_FINAL)
    log.info(f"Loaded {len(df)} samples | Slop: {(df['label']==1).sum()} | Genuine: {(df['label']==0).sum()}")

    # ── Xử lý missing values ─────────────────────────────────────────────────
    total_missing = 0
    
    for col, feature_name in MISSING_FEATURES.items():
        if col not in df.columns:
            continue
            
        missing_count = 0
        for label_val in [LABEL_SLOP, LABEL_GENUINE]:
            mask_missing = (df[col] < 0) & (df["label"] == label_val)
            mask_valid   = (df[col] >= 0) & (df["label"] == label_val)
            
            if mask_missing.sum() > 0 and mask_valid.sum() > 0:
                median_val = df.loc[mask_valid, col].median()
                df.loc[mask_missing, col] = median_val
                missing_count += mask_missing.sum()
                log.info(f"Imputed {mask_missing.sum()} missing {feature_name} values "
                        f"(label={label_val}) với median={median_val:.4f}")
        
        total_missing += missing_count
    
    if total_missing > 0:
        log.info(f"Total missing imputed: {total_missing} values across {len(MISSING_FEATURES)} features")

    # Đảm bảo tất cả feature columns tồn tại
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        log.error(f"Thiếu columns trong CSV: {missing_cols}")
        log.error(f"Chạy lại features.py để tạo đủ features.")
        sys.exit(1)

    X = df[FEATURE_COLS].values
    y = df["label"].values

    log.info(f"Feature matrix shape: {X.shape}")
    log.info(f"Features used ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS[:5])}...")
    log.info(f"Feature list: {', '.join(FEATURE_COLS[-5:])}")

    return df, X, y


# ══════════════════════════════════════════════════════════════════════════════
# 2. Train
# ══════════════════════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """
    Pipeline: StandardScaler → RandomForest
    Scaler cần thiết vì features có range rất khác nhau.
    class_weight='balanced' tự động xử lý imbalance 142:62.
    """
    rf = RandomForestClassifier(
        n_estimators  = RF_N_ESTIMATORS,
        max_depth     = RF_MAX_DEPTH,
        class_weight  = "balanced",
        random_state  = RF_RANDOM_STATE,
        n_jobs        = -1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    rf),
    ])


def train(X: np.ndarray, y: np.ndarray) -> tuple[Pipeline, dict]:
    """
    Train/test split + cross-validation.
    Trả về (fitted_pipeline, metrics_dict).
    """
    # Stratified split giữ tỷ lệ label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = RF_TEST_SIZE,
        random_state = RF_RANDOM_STATE,
        stratify     = y,
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    log.info(f"Train label distribution — Slop: {(y_train==1).sum()} | Genuine: {(y_train==0).sum()}")

    # ── Cross-validation trên train set ──────────────────────────────────────
    log.info(f"Chạy {RF_CV_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=RF_CV_FOLDS, shuffle=True, random_state=RF_RANDOM_STATE)

    pipeline = build_pipeline()
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv      = cv,
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"],
        return_train_score = False,
    )

    cv_metrics = {
        "cv_accuracy" : cv_results["test_accuracy"].mean(),
        "cv_precision": cv_results["test_precision"].mean(),
        "cv_recall"   : cv_results["test_recall"].mean(),
        "cv_f1"       : cv_results["test_f1"].mean(),
        "cv_roc_auc"  : cv_results["test_roc_auc"].mean(),
        # Std để biết model có ổn định không
        "cv_f1_std"   : cv_results["test_f1"].std(),
        "cv_auc_std"  : cv_results["test_roc_auc"].std(),
    }

    log.info(f"CV F1: {cv_metrics['cv_f1']:.4f} (±{cv_metrics['cv_f1_std']:.4f})")
    log.info(f"CV ROC-AUC: {cv_metrics['cv_roc_auc']:.4f} (±{cv_metrics['cv_auc_std']:.4f})")

    # ── Train lại trên toàn bộ train set ─────────────────────────────────────
    pipeline.fit(X_train, y_train)

    # ── Đánh giá trên test set ────────────────────────────────────────────────
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "test_accuracy" : accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall"   : recall_score(y_test, y_pred),
        "test_f1"       : f1_score(y_test, y_pred),
        "test_roc_auc"  : roc_auc_score(y_test, y_pred_prob),
    }

    log.info(f"Test F1: {test_metrics['test_f1']:.4f} | Test ROC-AUC: {test_metrics['test_roc_auc']:.4f}")

    metrics = {**cv_metrics, **test_metrics,
               "y_test": y_test, "y_pred": y_pred,
               "X_train": X_train, "y_train": y_train}

    return pipeline, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 3. Report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(pipeline: Pipeline, metrics: dict, df: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("KẾT QUẢ TRAINING — ICF DETECTOR (15 features)")
    print("=" * 70)

    print(f"\n── {RF_CV_FOLDS}-Fold Cross-Validation (trên train set) ──")
    print(f"  Accuracy  : {metrics['cv_accuracy']:.4f}")
    print(f"  Precision : {metrics['cv_precision']:.4f}")
    print(f"  Recall    : {metrics['cv_recall']:.4f}")
    print(f"  F1-score  : {metrics['cv_f1']:.4f}  (±{metrics['cv_f1_std']:.4f})")
    print(f"  ROC-AUC   : {metrics['cv_roc_auc']:.4f}  (±{metrics['cv_auc_std']:.4f})")

    print(f"\n── Test Set ({int(RF_TEST_SIZE*100)}%) ──")
    print(f"  Accuracy  : {metrics['test_accuracy']:.4f}")
    print(f"  Precision : {metrics['test_precision']:.4f}")
    print(f"  Recall    : {metrics['test_recall']:.4f}")
    print(f"  F1-score  : {metrics['test_f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['test_roc_auc']:.4f}")

    print(f"\n── Confusion Matrix ──")
    cm = confusion_matrix(metrics["y_test"], metrics["y_pred"])
    tn, fp, fn, tp = cm.ravel()
    print(f"  {'':20} Predicted Genuine  Predicted Slop")
    print(f"  {'Actual Genuine':20} {tn:^17} {fp:^14}")
    print(f"  {'Actual Slop':20} {fn:^17} {tp:^14}")
    print(f"\n  ✓ True Negatives  (TN): {tn:2d}  — Genuine được nhận diện đúng")
    print(f"  ✓ True Positives  (TP): {tp:2d}  — Slop bị bắt đúng")
    print(f"  ✗ False Positives (FP): {fp:2d}  — Genuine bị nhầm thành Slop (false alarm)")
    print(f"  ✗ False Negatives (FN): {fn:2d}  — Slop bị sót (miss)")

    print(f"\n── Classification Report ──")
    print(classification_report(
        metrics["y_test"], metrics["y_pred"],
        target_names=["Genuine (0)", "Slop (1)"]
    ))

    # Feature importance
    rf_model = pipeline.named_steps["clf"]
    importances = rf_model.feature_importances_
    
    print(f"\n── Feature Importance (Random Forest) ──")
    print(f"  {'Feature':<30} {'Importance':>10}  {'Bar'}")
    print(f"  {'─'*30} {'─'*10}  {'─'*40}")
    
    fi_pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in fi_pairs:
        bar = "█" * int(imp * 50)
        print(f"  {feat:<30} {imp:>10.4f}  {bar}")
    
    # Highlight top 3 features
    print(f"\n  🔥 Top 3 features:")
    for i, (feat, imp) in enumerate(fi_pairs[:3], 1):
        print(f"     {i}. {feat} ({imp:.4f})")

    # Hiển thị vị trí của các feature mới đáng chú ý
    highlight_features = {
        "view_per_video": "Lượt xem trung bình mỗi video",
        "sub_to_view_velocity_ratio": "Tỷ lệ tăng trưởng sub/view",
        "subscriber_velocity": "Tốc độ tăng sub",
        "video_upload_frequency": "Tần suất đăng video",
        "if_anomaly_score": "Anomaly score (Isolation Forest)",
    }
    for feat, label in highlight_features.items():
        imp_val = next((imp for f, imp in fi_pairs if f == feat), None)
        if imp_val is not None:
            rank = next(i for i, (f, _) in enumerate(fi_pairs, 1) if f == feat)
            print(f"     ⋮")
            print(f"     {rank}. {feat} ({imp_val:.4f}) — {label}")

    print("=" * 70)

    # Cảnh báo
    if metrics["test_f1"] > 0.98:
        print("\n⚠️  F1 > 0.98 — Kiểm tra overfitting! Dataset có thể quá nhỏ hoặc features quá dễ.")
    if metrics["cv_f1_std"] > 0.1:
        print(f"\n⚠️  CV F1 std={metrics['cv_f1_std']:.3f} cao — Model không ổn định, cần thêm data.")
    
    f1_gap = abs(metrics["cv_f1"] - metrics["test_f1"])
    if f1_gap > 0.1:
        print(f"\n⚠️  Gap CV-Test F1={f1_gap:.3f} lớn — Có thể overfitting hoặc test set không đại diện.")
    
    print(f"\n💡 Model hiện tại ({len(FEATURE_COLS)} features) gồm 4 nhóm:")
    print("   📈 Nhóm chuỗi thời gian & vận tốc đăng bài:")
    print("       time_interval_std, upload_burst_ratio, video_upload_frequency, if_anomaly_score, view_per_video")
    print("   📝 Nhóm dấu vết cấu trúc & định dạng văn bản AI:")
    print("       dash_density, title_length_std, capitalization_ratio, opening_repeat_ratio, temporal_clickbait_ratio")
    print("   🔄 Nhóm độ đa dạng & tương đồng nội dung:")
    print("       type_token_ratio, avg_title_similarity")
    print("   💰 Nhóm chỉ số tài chính & gian lận tương tác:")
    print("       sub_to_view_ratio, subscriber_velocity, sub_to_view_velocity_ratio")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Save
# ══════════════════════════════════════════════════════════════════════════════

def save_model(pipeline: Pipeline) -> None:
    Paths.MODELS.mkdir(parents=True, exist_ok=True)

    # Save pipeline (scaler + rf bundled)
    model_data = {
        "pipeline": pipeline,
        "feature_cols": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
    }
    
    with open(Paths.MODEL_RF, "wb") as f:
        pickle.dump(model_data, f)

    log.info(f"Model saved → {Paths.MODEL_RF}")
    log.info(f"  Features ({len(FEATURE_COLS)}): {', '.join(FEATURE_COLS[:5])}...")
    log.info(f"  Pipeline: StandardScaler → RandomForest(n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH})")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train ICF detector model")
    parser.add_argument("--no-save", action="store_true",
                        help="Chạy train nhưng không lưu model (dùng để test)")
    args = parser.parse_args()

    validate()

    if not Paths.FEATURES_FINAL.exists():
        log.error(f"Không tìm thấy {Paths.FEATURES_FINAL}. Hãy chạy features.py trước.")
        sys.exit(1)

    # Load
    log.info("=" * 70)
    log.info("BẮT ĐẦU TRAINING")
    log.info("=" * 70)
    df, X, y = load_data()

    # Train
    pipeline, metrics = train(X, y)

    # Report
    print_report(pipeline, metrics, df)

    # Save
    if not args.no_save:
        save_model(pipeline)
        print(f"\n✅ Model đã được lưu tại: {Paths.MODEL_RF}")
    else:
        log.info("\n--no-save: Model không được lưu.")


if __name__ == "__main__":
    main()