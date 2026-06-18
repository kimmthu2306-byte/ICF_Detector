"""
anomaly.py — Isolation Forest trên upload interval series
==========================================================
Huấn luyện Isolation Forest trên toàn bộ khoảng cách đăng bài (flatten)
của tất cả kênh, sau đó tính điểm bất thường trung bình cho mỗi kênh.

Cách dùng trong features.py:
    from src.anomaly import compute_interval_anomaly_scores

    df_feat["interval_anomaly_score"] = compute_interval_anomaly_scores(
        df_raw["video_timestamps"]
    )

Hàm này sẽ tự động huấn luyện model và trả về mảng scores cho từng kênh.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    IF_CONTAMINATION,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    Paths,
)
from src.utils import _parse_timestamps  # sửa import từ utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def extract_all_intervals(timestamps_series: pd.Series) -> np.ndarray:
    """
    Trích xuất tất cả khoảng cách đăng bài (ngày) từ các chuỗi timestamp của tất cả kênh.
    Trả về mảng 1D các intervals (bỏ qua các interval > 365 ngày).
    """
    all_intervals = []

    for ts_json in timestamps_series:
        dts = _parse_timestamps(ts_json)
        if len(dts) < 2:
            continue
        dts.sort()
        intervals = [
            (dts[i + 1] - dts[i]).total_seconds() / 86400
            for i in range(len(dts) - 1)
        ]
        # Lọc outlier quá lớn (có thể là kênh bỏ lâu rồi quay lại)
        intervals = [x for x in intervals if x <= 365]
        all_intervals.extend(intervals)

    return np.array(all_intervals).reshape(-1, 1)


def train_interval_if(intervals: np.ndarray) -> Pipeline:
    """
    Huấn luyện Isolation Forest trên dữ liệu interval.
    Trả về Pipeline (StandardScaler + IsolationForest).
    """
    if len(intervals) < 10:
        log.warning("Quá ít intervals để huấn luyện Isolation Forest (cần ≥10).")
        # Trả về pipeline rỗng (sẽ trả về score 0 cho mọi kênh)
        return None

    scaler = StandardScaler()
    if_model = IsolationForest(
        contamination=IF_CONTAMINATION,
        n_estimators=IF_N_ESTIMATORS,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline([("scaler", scaler), ("if", if_model)])
    pipeline.fit(intervals)

    log.info(f"Isolation Forest trained on {len(intervals)} intervals.")
    return pipeline


def compute_channel_scores(
    timestamps_series: pd.Series,
    pipeline: Optional[Pipeline] = None,
) -> np.ndarray:
    """
    Với mỗi kênh, tính điểm bất thường trung bình của các interval.
    Nếu pipeline là None, trả về mảng 0.
    """
    if pipeline is None:
        return np.zeros(len(timestamps_series))

    scores = []
    for ts_json in timestamps_series:
        dts = _parse_timestamps(ts_json)
        if len(dts) < 2:
            scores.append(0.0)
            continue
        dts.sort()
        intervals = [
            (dts[i + 1] - dts[i]).total_seconds() / 86400
            for i in range(len(dts) - 1)
        ]
        intervals = [x for x in intervals if x <= 365]
        if not intervals:
            scores.append(0.0)
            continue

        X = np.array(intervals).reshape(-1, 1)
        # Lấy raw anomaly scores (càng âm càng bất thường)
        raw_scores = pipeline.named_steps["if"].score_samples(
            pipeline.named_steps["scaler"].transform(X)
        )
        # Đảo dấu để dễ hiểu: điểm cao = bất thường
        # Nhân -1 và chuẩn hóa về [0,1] dựa trên min/max toàn cục đã biết từ train?
        # Thay vì chuẩn hóa, ta dùng trung bình raw_scores đã đảo dấu
        # Để tránh phụ thuộc vào thang đo, ta dùng percentile hoặc trung bình đảo dấu.
        # Ở đây dùng trung bình -raw_scores (càng lớn càng bất thường)
        mean_score = float(np.mean(-raw_scores))
        scores.append(mean_score)

    return np.array(scores)


def compute_interval_anomaly_scores(
    timestamps_series: pd.Series,
    retrain: bool = True,
) -> np.ndarray:
    """
    Hàm chính: huấn luyện (hoặc load model) và trả về điểm bất thường cho từng kênh.
    Nếu retrain=True, huấn luyện lại từ dữ liệu hiện có.
    Nếu False, có thể load model từ file (chưa implement).

    Trả về mảng scores cùng chiều với timestamps_series.
    """
    if retrain:
        intervals = extract_all_intervals(timestamps_series)
        pipeline = train_interval_if(intervals)
    else:
        # TODO: load pipeline từ file
        pipeline = None
        log.warning("Chưa hỗ trợ load model, đang huấn luyện lại.")
        intervals = extract_all_intervals(timestamps_series)
        pipeline = train_interval_if(intervals)

    return compute_channel_scores(timestamps_series, pipeline)


# ══════════════════════════════════════════════════════════════════════════════
# Chạy thử (nếu là script chính)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Đọc channels_raw.csv và tính scores, lưu thành file để tham khảo
    csv_path = Paths.CHANNELS_RAW
    if not csv_path.exists():
        log.error(f"Không tìm thấy {csv_path}")
        sys.exit(1)

    df_raw = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df_raw)} channels.")
    scores = compute_interval_anomaly_scores(df_raw["video_timestamps"], retrain=True)
    df_raw["interval_anomaly_score"] = scores
    out_path = Paths.PROCESSED / "interval_anomaly_scores.csv"
    df_raw[["channel_id", "title", "label", "interval_anomaly_score"]].to_csv(
        out_path, index=False, encoding="utf-8"
    )
    log.info(f"Saved interval anomaly scores to {out_path}")