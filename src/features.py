"""
features.py — Tính 15 đặc trưng từ channels_raw.csv
=====================================================
Input  : data/collected/channels_raw.csv
Output : data/processed/features_final.csv

15 features (đã loại bỏ trùng lặp, thêm feature mới):
  Nhóm Chuỗi thời gian & Vận tốc đăng bài (5):
    time_interval_std        — Độ lệch chuẩn khoảng cách đăng bài (ngày)
    upload_burst_ratio       — Tỷ lệ video đăng dồn dập (< 30% median interval)
    video_upload_frequency   — Tần suất đăng video (videos/ngày)
    if_anomaly_score         — Isolation Forest anomaly score (toàn bộ intervals)
    view_per_video           — Lượt xem trung bình mỗi video (view_count / video_count)

  Nhóm Dấu vết cấu trúc & Định dạng văn bản AI (5):
    dash_density             — Mật độ dấu gạch ngang (— hoặc --) trong title
    title_length_std         — Std độ dài title (thấp → dùng template cứng)
    capitalization_ratio     — Mật độ chữ viết hoa trong title
    opening_repeat_ratio     — Tỷ lệ lặp cụm 3-từ mở đầu
    temporal_clickbait_ratio — Tỷ lệ title có ?/!/số × time_interval (lai)

  Nhóm Độ đa dạng & Tương đồng nội dung (2):
    type_token_ratio         — Độ đa dạng từ vựng (unique/total words)
    avg_title_similarity     — Độ tương đồng nội tại giữa các title

  Nhóm Chỉ số tài chính & Gian lận tương tác (3):
    sub_to_view_ratio        — Sub / View (sub cao bất thường → mua sub)
    subscriber_velocity      — Tốc độ tăng sub (subs/ngày)
    sub_to_view_velocity_ratio — log10((sub_vel+1)/(view_vel+1)) – phát hiện mua sub

Cách chạy:
    python src/features.py
    python src/features.py --verbose   # in chi tiết từng kênh
"""

import sys
import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    SIMILARITY_WINDOW,
    TFIDF_MAX_FEATURES,
    IF_CONTAMINATION,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    MIN_UPLOAD_INTERVAL_DAYS,
    Paths,
    validate,
)
# Import từ anomaly.py
from src.anomaly import compute_interval_anomaly_scores
# Import helper từ utils.py
from src.utils import _parse_timestamps, _parse_titles, _flatten_titles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Từ điển pattern (giữ lại F3 clickbait_freq làm feature phụ để so sánh)
# ══════════════════════════════════════════════════════════════════════════════

# F3 — Clickbait / AI template phrases (EN + VI) — feature phụ, giữ lại để tham khảo
CLICKBAIT_PATTERNS = [
    # English
    r"(?i)\byou (won'?t|will never) believe\b",
    r"(?i)\bthe (dark |hidden |shocking |untold |real |secret )?truth\b",
    r"(?i)\bwhat (nobody|no one) tells you\b",
    r"(?i)\bchanged (everything|the world|history)\b",
    r"(?i)\bmost people don'?t know\b",
    r"(?i)\bgenius of\b",
    r"(?i)\bexplained (simply|in \d+ minutes?)\b",
    r"(?i)\bthis (changes|will change) everything\b",
    r"(?i)\bthe (mind|brain) of\b",
    r"(?i)\bfeynman (explains|on|about|method)\b",
    r"(?i)\beinstein (explains|on|about)\b",
    r"(?i)\bcarl sagan (on|about|explains)\b",
    r"(?i)\bnot .{1,30} but\b",           # "not X but Y" structure
    r"(?i)\bfew people (know|realize)\b",
    # Vietnamese
    r"ít ai biết",
    r"sự thật (kinh hoàng|bí ẩn|đáng sợ|không ngờ)",
    r"không phải .{1,20} mà là",
    r"(bí mật|bí ẩn) (đằng sau|của|về)",
    r"tại sao (không ai|ít người)",
    r"review phim",   # mass-produced film review channels
]

# F2 — Dash patterns
DASH_PATTERN = re.compile(r"—|--|–")


def _get_channel_age_days(timestamps_json: str) -> float:
    """
    Tính tuổi kênh (ngày) từ timestamp cũ nhất đến hiện tại.
    Trả về -1 nếu không parse được timestamps.
    """
    dts = _parse_timestamps(timestamps_json)
    if not dts:
        return -1.0

    first_video_date = min(dts)
    now = datetime.now(timezone.utc)
    channel_age_days = (now - first_video_date).total_seconds() / 86400

    return max(channel_age_days, 1.0)  # Tối thiểu 1 ngày


# ══════════════════════════════════════════════════════════════════════════════
# Hàm tính các feature
# ══════════════════════════════════════════════════════════════════════════════

def compute_time_interval_std(timestamps_json: str) -> float:
    """F1: Std khoảng cách đăng bài (ngày)."""
    dts = _parse_timestamps(timestamps_json)
    if len(dts) < 3:
        return -1.0
    dts.sort()
    intervals = [
        (dts[i+1] - dts[i]).total_seconds() / 86400
        for i in range(len(dts) - 1)
    ]
    intervals = [x for x in intervals if x <= 365]
    if len(intervals) < 2:
        return -1.0
    return float(np.std(intervals))


def compute_upload_burst_ratio(timestamps_json: str) -> float:
    """F6: Tỷ lệ video đăng dồn dập."""
    dts = _parse_timestamps(timestamps_json)
    if len(dts) < 3:
        return 0.0
    dts.sort()
    intervals = [
        (dts[i+1] - dts[i]).total_seconds() / 86400
        for i in range(len(dts) - 1)
    ]
    intervals = [x for x in intervals if x <= 365]
    if not intervals:
        return 0.0
    threshold = np.median(intervals) * 0.3
    burst_count = sum(1 for x in intervals if x < threshold)
    return burst_count / len(intervals)


def compute_video_upload_frequency(timestamps_json: str, video_count: int = None, n_videos_crawled: int = None) -> float:
    """F14: Tần suất đăng video (videos/ngày)."""
    dts = _parse_timestamps(timestamps_json)
    if not dts:
        return -1.0
    first_video_date = min(dts)
    now = datetime.now(timezone.utc)
    channel_age_days = (now - first_video_date).total_seconds() / 86400
    num_videos = n_videos_crawled if n_videos_crawled and n_videos_crawled > 0 else (
        video_count if video_count and video_count > 0 else len(dts)
    )
    if channel_age_days < 1:
        channel_age_days = 1.0
    return num_videos / channel_age_days


def compute_view_per_video(view_count: int, video_count: int) -> float:
    """Mới: Lượt xem trung bình mỗi video."""
    try:
        views = float(view_count) if view_count is not None else 0.0
        videos = float(video_count) if video_count is not None else 0.0
        if videos <= 0:
            return 0.0
        return views / videos
    except (ValueError, TypeError):
        return 0.0


def compute_subscriber_velocity(timestamps_json: str, subscriber_count: int) -> float:
    """F17: Tốc độ tăng sub (subs/ngày)."""
    channel_age_days = _get_channel_age_days(timestamps_json)
    if channel_age_days < 0:
        return -1.0
    try:
        subs = float(subscriber_count) if subscriber_count is not None else 0.0
    except (ValueError, TypeError):
        return -1.0
    if subs <= 0:
        return 0.0
    return subs / channel_age_days


def compute_sub_to_view_velocity_ratio(subscriber_velocity: float, view_velocity: float) -> float:
    """Mới: log10((sub_vel+1)/(view_vel+1))."""
    # Nếu view_velocity không tính được (<=0) thì đặt về 0?
    if view_velocity <= 0 or subscriber_velocity <= 0:
        return 0.0
    return float(np.log10((subscriber_velocity + 1) / (view_velocity + 1)))


def compute_dash_density(titles_json: str) -> float:
    """F2: Mật độ dấu gạch ngang."""
    text = _flatten_titles(titles_json)
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    n_dashes = len(DASH_PATTERN.findall(text))
    return n_dashes / len(words)


def compute_title_length_std(titles_json: str) -> float:
    """F8: Std độ dài title."""
    titles = _parse_titles(titles_json)
    lengths = [len(t) for t in titles]
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths))


def compute_capitalization_ratio(titles_json: str) -> float:
    """F10: Mật độ chữ viết hoa."""
    titles = _parse_titles(titles_json)
    if not titles:
        return 0.0
    ratios = []
    for title in titles:
        letters = [c for c in title if c.isalpha()]
        if not letters:
            continue
        uppercase = sum(1 for c in letters if c.isupper())
        ratios.append(uppercase / len(letters))
    return float(np.mean(ratios)) if ratios else 0.0


def compute_opening_repeat_ratio(titles_json: str, n_gram: int = 3) -> float:
    """F11: Tỷ lệ lặp cụm mở đầu."""
    titles = _parse_titles(titles_json)
    if len(titles) < 2:
        return 0.0
    openings = []
    for title in titles:
        words = title.lower().split()[:n_gram]
        if words:
            openings.append(" ".join(words))
    if not openings:
        return 0.0
    unique_openings = len(set(openings))
    total_titles = len(openings)
    return 1.0 - (unique_openings / total_titles)


def compute_temporal_clickbait_ratio(titles_json: str, time_interval_std: float) -> float:
    """F12: Tỷ lệ title có ?/!/số × time_interval."""
    titles = _parse_titles(titles_json)
    if not titles:
        return 0.0
    clickbait_markers = 0
    for title in titles:
        if re.search(r"[?!]", title) or re.search(r"\d+", title):
            clickbait_markers += 1
    ratio = clickbait_markers / len(titles)
    if time_interval_std < 0:
        return 0.0
    return ratio / (1.0 + time_interval_std)


def compute_type_token_ratio(titles_json: str) -> float:
    """F4: Độ đa dạng từ vựng."""
    text = _flatten_titles(titles_json)
    if not text:
        return 0.0
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_avg_title_similarity(titles_json: str, window: int = SIMILARITY_WINDOW) -> float:
    """F5: Trung bình cosine similarity giữa các title."""
    titles = _parse_titles(titles_json)
    titles = titles[:window]
    if len(titles) < 2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(titles)
        sim_matrix = cosine_similarity(tfidf_matrix)
        n = sim_matrix.shape[0]
        upper = [sim_matrix[i, j] for i in range(n) for j in range(i+1, n)]
        return float(np.mean(upper)) if upper else 0.0
    except Exception:
        return 0.0


def compute_sub_to_view_ratio(subscriber_count, view_count) -> float:
    """F7: Sub / View."""
    try:
        sub = float(subscriber_count) if subscriber_count is not None else 0.0
        view = float(view_count) if view_count is not None else 0.0
        return sub / (view + 1)
    except (ValueError, TypeError):
        return 0.0


def compute_view_velocity(timestamps_json: str, view_count: int) -> float:
    """(tạm) Tính view_velocity để dùng cho sub_to_view_velocity_ratio."""
    channel_age_days = _get_channel_age_days(timestamps_json)
    if channel_age_days < 0:
        return -1.0
    try:
        views = float(view_count) if view_count is not None else 0.0
    except (ValueError, TypeError):
        return -1.0
    if views <= 0:
        return 0.0
    return views / channel_age_days


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline chính
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df_raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Nhận channels_raw DataFrame, trả về features DataFrame với 15 features.
    """
    log.info(f"Bắt đầu tính features cho {len(df_raw)} kênh...")
    rows = []

    for _, row in df_raw.iterrows():
        # --- Tính các feature cơ bản ---
        f1_time_std = compute_time_interval_std(row["video_timestamps"])
        f6_burst = compute_upload_burst_ratio(row["video_timestamps"])
        f14_freq = compute_video_upload_frequency(
            row["video_timestamps"],
            row.get("video_count"),
            row.get("n_videos_crawled")
        )

        # --- Text features ---
        f2_dash = compute_dash_density(row["video_titles"])
        f8_title_std = compute_title_length_std(row["video_titles"])
        f10_cap = compute_capitalization_ratio(row["video_titles"])
        f11_open = compute_opening_repeat_ratio(row["video_titles"])
        f12_temp_click = compute_temporal_clickbait_ratio(row["video_titles"], f1_time_std)

        f4_ttr = compute_type_token_ratio(row["video_titles"])
        f5_sim = compute_avg_title_similarity(row["video_titles"])

        f7_sub_view = compute_sub_to_view_ratio(
            row.get("subscriber_count", 0),
            row.get("view_count", 0)
        )

        # --- Velocity features ---
        # Tính view_velocity và subscriber_velocity (dùng cho cả feature mới)
        view_vel = compute_view_velocity(row["video_timestamps"], row.get("view_count"))
        sub_vel = compute_subscriber_velocity(row["video_timestamps"], row.get("subscriber_count"))

        # Feature mới: view_per_video
        view_per_vid = compute_view_per_video(row.get("view_count"), row.get("video_count"))

        # Feature mới: sub_to_view_velocity_ratio
        sub_view_vel_ratio = compute_sub_to_view_velocity_ratio(sub_vel, view_vel)

        # --- Gộp các feature cần output ---
        feat_row = {
            "channel_id": row["channel_id"],
            "title": row["title"],
            "label": row["label"],

            # Nhóm chuỗi thời gian & vận tốc đăng bài
            "time_interval_std": f1_time_std,
            "upload_burst_ratio": f6_burst,
            "video_upload_frequency": f14_freq,
            "view_per_video": view_per_vid,
            # if_anomaly_score sẽ được thêm sau

            # Nhóm dấu vết cấu trúc & định dạng văn bản AI
            "dash_density": f2_dash,
            "title_length_std": f8_title_std,
            "capitalization_ratio": f10_cap,
            "opening_repeat_ratio": f11_open,
            "temporal_clickbait_ratio": f12_temp_click,

            # Nhóm độ đa dạng & tương đồng nội dung
            "type_token_ratio": f4_ttr,
            "avg_title_similarity": f5_sim,

            # Nhóm chỉ số tài chính & gian lận tương tác
            "sub_to_view_ratio": f7_sub_view,
            "subscriber_velocity": sub_vel,
            "sub_to_view_velocity_ratio": sub_view_vel_ratio,

            # Meta (chỉ để tham khảo, không dùng trong train)
            "subscriber_count": row.get("subscriber_count", 0),
            "view_count": row.get("view_count", 0),
            "video_count": row.get("video_count", 0),
            "n_videos_crawled": row.get("n_videos_crawled", 0),
        }
        rows.append(feat_row)

        if verbose:
            label_str = "SLOP" if row["label"] == 1 else "GENUINE"
            log.info(
                f"  [{label_str}] {row['title'][:35]:<35} "
                f"std={f1_time_std:6.2f} burst={f6_burst:.3f} freq={f14_freq:.2f} "
                f"view/vid={view_per_vid:.1f} sub_vel={sub_vel:.1f} ratio={sub_view_vel_ratio:.3f}"
            )

    df_feat = pd.DataFrame(rows)

    # ── Thêm Isolation Forest anomaly score dựa trên toàn bộ intervals ──
    log.info("Tính anomaly score bằng Isolation Forest trên toàn bộ intervals...")
    anomaly_scores = compute_interval_anomaly_scores(df_raw["video_timestamps"], retrain=True)
    df_feat["if_anomaly_score"] = anomaly_scores
    log.info(f"Anomaly score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")

    # Đảm bảo cột if_anomaly_score nằm ngay sau video_upload_frequency (theo nhóm)
    # Chúng ta có thể sắp xếp lại thứ tự cột nếu muốn, nhưng không bắt buộc.

    return df_feat


def print_summary(df: pd.DataFrame) -> None:
    """In tóm tắt phân phối feature theo label."""
    feature_cols = [
        "time_interval_std", "upload_burst_ratio", "video_upload_frequency",
        "if_anomaly_score", "view_per_video",
        "dash_density", "title_length_std", "capitalization_ratio",
        "opening_repeat_ratio", "temporal_clickbait_ratio",
        "type_token_ratio", "avg_title_similarity",
        "sub_to_view_ratio", "subscriber_velocity", "sub_to_view_velocity_ratio"
    ]

    print("\n" + "=" * 70)
    print("PHÂN PHỐI FEATURE THEO LABEL (15 features)")
    print("=" * 70)

    for col in feature_cols:
        slop_vals = df.loc[df["label"] == 1, col]
        genuine_vals = df.loc[df["label"] == 0, col]

        # Loại -1 (missing) khi tính mean
        slop_mean = slop_vals[slop_vals >= 0].mean()
        genuine_mean = genuine_vals[genuine_vals >= 0].mean()

        direction = "↑slop" if slop_mean > genuine_mean else "↓slop"
        print(
            f"  {col:<28} "
            f"Slop={slop_mean:10.4f}  Genuine={genuine_mean:10.4f}  {direction}"
        )

    print()
    print(f"  Kênh thiếu data timestamps (F1=-1): "
          f"{(df['time_interval_std'] < 0).sum()} kênh")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compute features from channels_raw.csv")
    parser.add_argument("--verbose", action="store_true",
                        help="In chi tiết feature của từng kênh")
    args = parser.parse_args()

    validate()

    if not Paths.CHANNELS_RAW.exists():
        log.error(f"Không tìm thấy {Paths.CHANNELS_RAW}. Hãy chạy crawl.py trước.")
        sys.exit(1)

    log.info(f"Đọc {Paths.CHANNELS_RAW.name}...")
    df_raw = pd.read_csv(Paths.CHANNELS_RAW)
    log.info(f"  {len(df_raw)} kênh | "
             f"Slop: {(df_raw['label']==1).sum()} | "
             f"Genuine: {(df_raw['label']==0).sum()}")

    df_feat = build_features(df_raw, verbose=args.verbose)

    # Lưu
    Paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(Paths.FEATURES_FINAL, index=False, encoding="utf-8")
    log.info(f"Saved → {Paths.FEATURES_FINAL}")
    log.info(f"Shape: {df_feat.shape} | Columns: {list(df_feat.columns)}")

    print_summary(df_feat)


if __name__ == "__main__":
    main()