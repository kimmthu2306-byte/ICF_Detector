"""
crawl_expand.py — Cào thêm dữ liệu từ các file expand
====================================================
Input: data/raw/AI slop expand.txt và data/raw/non AI expand.txt
Output: cập nhật data/collected/channels_raw.csv
Cách chạy: python src/crawl_expand.py
"""

import sys
import time
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Paths, LABEL_SLOP, LABEL_GENUINE, validate
from src.crawl import build_youtube, crawl_channel, load_seed_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Đường dẫn đến thư mục chứa seed files
RAW_DIR = Paths.SEED_SLOP.parent

EXPAND_SLOP = RAW_DIR / "AI slop expand.txt"
EXPAND_GENUINE = RAW_DIR / "non AI expand.txt"
OUTPUT_CSV = Paths.CHANNELS_RAW


def get_existing_ids(csv_path: Path) -> set:
    """Trả về set các channel_id đã có trong CSV (để skip)."""
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(df["channel_id"].dropna().astype(str))


def main():
    validate()

    # Kiểm tra file expand tồn tại
    if not EXPAND_SLOP.exists():
        log.error(f"Không tìm thấy {EXPAND_SLOP}")
        sys.exit(1)
    if not EXPAND_GENUINE.exists():
        log.error(f"Không tìm thấy {EXPAND_GENUINE}")
        sys.exit(1)

    # Đọc seed từ hai file
    slop_seeds = load_seed_file(EXPAND_SLOP, label=LABEL_SLOP)
    genuine_seeds = load_seed_file(EXPAND_GENUINE, label=LABEL_GENUINE)
    all_seeds = slop_seeds + genuine_seeds
    log.info(f"Tổng số seed mới: {len(all_seeds)} (slop={len(slop_seeds)}, genuine={len(genuine_seeds)})")

    # Lấy danh sách ID đã crawl
    existing_ids = get_existing_ids(OUTPUT_CSV)
    log.info(f"Số kênh đã crawl hiện có: {len(existing_ids)}")

    # Đọc CSV cũ nếu có
    if OUTPUT_CSV.exists():
        df_old = pd.read_csv(OUTPUT_CSV)
        rows = df_old.to_dict("records")
    else:
        rows = []

    yt = build_youtube()
    new_count = 0

    for seed in all_seeds:
        row = crawl_channel(yt, seed)
        if row is None:
            log.warning(f"Bỏ qua seed {seed['raw_url']} (crawl thất bại)")
            continue

        if row["channel_id"] in existing_ids:
            log.info(f"Bỏ qua {row['title']} ({row['channel_id']}) - đã có")
            continue

        rows.append(row)
        existing_ids.add(row["channel_id"])
        new_count += 1
        log.info(f"Đã thêm {row['title']} ({row['channel_id']})")

        # Checkpoint mỗi 10 kênh
        if new_count % 10 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
            log.info(f"Checkpoint: đã lưu {len(rows)} kênh vào {OUTPUT_CSV}")

        time.sleep(0.5)

    # Lưu lần cuối
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"Hoàn tất! Đã thêm {new_count} kênh mới. Tổng {len(rows)} kênh trong {OUTPUT_CSV}")


if __name__ == "__main__":
    main()