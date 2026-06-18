"""
crawl.py — Thu thập metadata kênh YouTube từ seed files
=========================================================
Input  : data/raw/AI slop.txt
         data/raw/non AI.txt
Output : data/collected/channels_raw.csv
 
Mỗi hàng trong CSV = 1 kênh với:
  - Thông tin cơ bản (tên, ngày tạo, subscriber, v.v.)
  - Danh sách timestamps của N video gần nhất (để tính time-series features)
  - Label gốc (1 = slop, 0 = genuine)
 
Cách chạy:
    python src/crawl.py
    python src/crawl.py --limit 10   # test với 10 kênh đầu mỗi loại
"""
 
import argparse
import sys
import time
import json
import re
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import unquote
 
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
 
# Import config (chạy từ thư mục gốc project)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    YOUTUBE_API_KEY,
    VIDEOS_PER_CHANNEL,
    Paths,
    LABEL_SLOP,
    LABEL_GENUINE,
    validate,
)
 
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. Đọc và parse seed files
# ══════════════════════════════════════════════════════════════════════════════
 
def parse_url(url: str) -> Optional[dict]:
    """
    Nhận một URL YouTube, trả về dict {"type": ..., "value": ...} để resolve
    thành channel_id qua API.
 
    Hỗ trợ:
      - https://www.youtube.com/@Handle        → type="handle"
      - https://www.youtube.com/channel/UC...  → type="id"
      - URL search (results?...) → None (bỏ qua)
    """
    url = unquote(url.strip())   # decode %C4%90 → Đ, v.v.
    if not url or url.startswith("#"):
        return None
 
    # Skip search result URLs
    if "results?" in url or "search_query" in url:
        log.debug(f"Skip search URL: {url}")
        return None
 
    # @Handle
    handle_match = re.search(r"youtube\.com/@([\w\-\.]+)", url)
    if handle_match:
        return {"type": "handle", "value": handle_match.group(1)}
 
    # /channel/UCxxxx
    id_match = re.search(r"youtube\.com/channel/(UC[\w\-]+)", url)
    if id_match:
        return {"type": "id", "value": id_match.group(1)}
 
    log.warning(f"Không nhận dạng được URL: {url}")
    return None
 
 
def load_seed_file(path: Path, label: int) -> list[dict]:
    """
    Đọc file txt chứa URLs, trả về list dict {"parsed": ..., "label": ..., "raw_url": ...}
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    seeds = []
    seen = set()
 
    for line in lines:
        url = line.strip().rstrip("\r")
        if not url:
            continue
        parsed = parse_url(url)
        if parsed is None:
            continue
 
        # Deduplicate
        key = f"{parsed['type']}:{parsed['value'].lower()}"
        if key in seen:
            continue
        seen.add(key)
 
        seeds.append({"parsed": parsed, "label": label, "raw_url": url})
 
    log.info(f"Đọc {len(seeds)} URL hợp lệ từ {path.name}")
    return seeds
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. YouTube API helpers
# ══════════════════════════════════════════════════════════════════════════════
 
def build_youtube():
    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
 
 
def resolve_channel_id(yt, parsed: dict) -> Optional[str]:
    """
    Chuyển @handle hoặc channel/ID thành channel_id chuẩn (UCxxxx).
    """
    if parsed["type"] == "id":
        return parsed["value"]
 
    # @handle → cần gọi API
    try:
        resp = yt.channels().list(
            part="id",
            forHandle=parsed["value"],
        ).execute()
        items = resp.get("items", [])
        if items:
            return items[0]["id"]
    except HttpError as e:
        log.warning(f"Không resolve được @{parsed['value']}: {e}")
    return None
 
 
def get_channel_info(yt, channel_id: str) -> Optional[dict]:
    """
    Lấy thông tin cơ bản của kênh:
      - title, description, publishedAt
      - subscriberCount, videoCount, viewCount
      - uploads playlist ID (để lấy video timestamps)
    """
    try:
        resp = yt.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id,
        ).execute()
        items = resp.get("items", [])
        if not items:
            return None
 
        ch = items[0]
        snippet = ch.get("snippet", {})
        stats   = ch.get("statistics", {})
        uploads_playlist = (
            ch.get("contentDetails", {})
              .get("relatedPlaylists", {})
              .get("uploads", "")
        )
 
        return {
            "channel_id"       : channel_id,
            "title"            : snippet.get("title", ""),
            "description"      : snippet.get("description", ""),
            "published_at"     : snippet.get("publishedAt", ""),
            "country"          : snippet.get("country", ""),
            "subscriber_count" : int(stats.get("subscriberCount", 0) or 0),
            "video_count"      : int(stats.get("videoCount", 0) or 0),
            "view_count"       : int(stats.get("viewCount", 0) or 0),
            "uploads_playlist" : uploads_playlist,
        }
    except HttpError as e:
        log.warning(f"get_channel_info({channel_id}) lỗi: {e}")
        return None
 
 
def get_video_timestamps(yt, uploads_playlist: str, max_results: int = VIDEOS_PER_CHANNEL) -> list[str]:
    """
    Lấy danh sách publishedAt của N video gần nhất trong playlist uploads.
    Trả về list ISO 8601 strings, từ mới → cũ.
    Quota cost: 1 unit/request, mỗi request lấy tối đa 50 video.
    """
    timestamps = []
    next_page_token = None
 
    try:
        while len(timestamps) < max_results:
            batch_size = min(50, max_results - len(timestamps))
            resp = yt.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist,
                maxResults=batch_size,
                pageToken=next_page_token,
            ).execute()
 
            for item in resp.get("items", []):
                ts = item["snippet"].get("publishedAt")
                if ts:
                    timestamps.append(ts)
 
            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break
 
    except HttpError as e:
        log.warning(f"get_video_timestamps({uploads_playlist}) lỗi: {e}")
 
    return timestamps
 
 
def get_video_titles(yt, uploads_playlist: str, max_results: int = VIDEOS_PER_CHANNEL) -> list[str]:
    """
    Lấy danh sách title của N video gần nhất.
    Dùng cho text similarity features trong features.py.
    """
    titles = []
    next_page_token = None
 
    try:
        while len(titles) < max_results:
            batch_size = min(50, max_results - len(titles))
            resp = yt.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist,
                maxResults=batch_size,
                pageToken=next_page_token,
            ).execute()
 
            for item in resp.get("items", []):
                title = item["snippet"].get("title", "")
                if title:
                    titles.append(title)
 
            next_page_token = resp.get("nextPageToken")
            if not next_page_token:
                break
 
    except HttpError as e:
        log.warning(f"get_video_titles({uploads_playlist}) lỗi: {e}")
 
    return titles
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. Pipeline chính
# ══════════════════════════════════════════════════════════════════════════════
 
def crawl_channel(yt, seed: dict) -> Optional[dict]:
    """
    Crawl toàn bộ thông tin 1 kênh từ seed dict.
    Trả về row dict để append vào CSV, hoặc None nếu thất bại.
    """
    parsed  = seed["parsed"]
    label   = seed["label"]
    raw_url = seed["raw_url"]
 
    # Bước 1: Resolve channel_id
    channel_id = resolve_channel_id(yt, parsed)
    if not channel_id:
        log.warning(f"  ✗ Không resolve được: {raw_url}")
        return None
 
    # Bước 2: Thông tin kênh
    info = get_channel_info(yt, channel_id)
    if not info:
        log.warning(f"  ✗ Không lấy được info: {channel_id}")
        return None
 
    # Bước 3: Video timestamps (cho time-series)
    timestamps = []
    titles     = []
    if info["uploads_playlist"]:
        timestamps = get_video_timestamps(yt, info["uploads_playlist"])
        titles     = get_video_titles(yt, info["uploads_playlist"])
 
    row = {
        **info,
        "label"              : label,
        "raw_url"            : raw_url,
        "n_videos_crawled"   : len(timestamps),
        # Lưu dưới dạng JSON string để fit vào 1 cell CSV
        # features.py sẽ parse lại
        "video_timestamps"   : json.dumps(timestamps),
        "video_titles"       : json.dumps(titles),
    }
 
    label_str = "SLOP" if label == LABEL_SLOP else "GENUINE"
    log.info(
        f"  ✓ [{label_str}] {info['title'][:40]:<40} "
        f"| {info['video_count']} videos | {len(timestamps)} timestamps"
    )
    return row
 
 
def crawl_all(seeds: list[dict], limit: Optional[int] = None) -> pd.DataFrame:
    """
    Crawl tất cả seed channels, lưu kết quả vào CSV (append theo batch để
    không mất data nếu bị gián đoạn giữa chừng).
    """
    yt = build_youtube()
 
    if limit:
        # Lấy `limit` kênh đầu của mỗi label
        slop    = [s for s in seeds if s["label"] == LABEL_SLOP][:limit]
        genuine = [s for s in seeds if s["label"] == LABEL_GENUINE][:limit]
        seeds   = slop + genuine
        log.info(f"[--limit] Chỉ crawl {len(seeds)} kênh ({limit} slop + {limit} genuine)")
 
    rows       = []
    failed     = []
    total      = len(seeds)
    output_csv = Paths.CHANNELS_RAW
 
    # Nếu đã có file từ lần chạy trước → load để skip các kênh đã crawl
    crawled_ids = set()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        try:
            existing = pd.read_csv(output_csv)
            crawled_ids = set(existing["channel_id"].dropna().tolist())
            log.info(f"Tìm thấy {len(crawled_ids)} kênh đã crawl trước đó → skip")
            rows = existing.to_dict("records")
        except pd.errors.EmptyDataError:
            log.warning("File CSV cũ bị rỗng → xóa và bắt đầu lại.")
            output_csv.unlink()
 
    log.info(f"Bắt đầu crawl {total} kênh...")
 
    for i, seed in enumerate(seeds, 1):
        log.info(f"[{i:>3}/{total}] {seed['raw_url']}")
 
        row = crawl_channel(yt, seed)
 
        if row is None:
            failed.append(seed["raw_url"])
            continue
 
        # Skip nếu đã có
        if row["channel_id"] in crawled_ids:
            log.info(f"  → Đã có trong CSV, skip.")
            continue
 
        rows.append(row)
        crawled_ids.add(row["channel_id"])
 
        # Save sau mỗi 10 kênh để tránh mất data
        if len(rows) % 10 == 0:
            _save_csv(rows, output_csv)
            log.info(f"  [checkpoint] Đã lưu {len(rows)} kênh vào {output_csv.name}")
 
        # Tránh rate limit: nghỉ 0.5s giữa các kênh
        time.sleep(0.5)
 
    # Save lần cuối
    _save_csv(rows, output_csv)
 
    # Summary
    log.info("=" * 60)
    log.info(f"Crawl xong!")
    log.info(f"  ✓ Thành công : {len(rows)} kênh")
    log.info(f"  ✗ Thất bại   : {len(failed)} kênh")
    if failed:
        log.info("  Danh sách thất bại:")
        for url in failed:
            log.info(f"    - {url}")
    log.info(f"  Saved → {output_csv}")
    log.info("=" * 60)
 
    return pd.read_csv(output_csv)
 
 
def _save_csv(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. Entrypoint
# ══════════════════════════════════════════════════════════════════════════════
 
def main():
    parser = argparse.ArgumentParser(description="Crawl YouTube channel metadata")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Giới hạn số kênh mỗi loại (dùng để test, ví dụ: --limit 5)"
    )
    args = parser.parse_args()
 
    # Kiểm tra config trước khi làm gì
    validate()
 
    # Load seeds
    slop_seeds    = load_seed_file(Paths.SEED_SLOP,    label=LABEL_SLOP)
    genuine_seeds = load_seed_file(Paths.SEED_GENUINE, label=LABEL_GENUINE)
    all_seeds     = slop_seeds + genuine_seeds
 
    # Crawl
    df = crawl_all(all_seeds, limit=args.limit)
 
    # Quick summary
    print("\n── Phân phối label trong CSV ──")
    print(df["label"].value_counts().rename({LABEL_SLOP: "Slop", LABEL_GENUINE: "Genuine"}))
    print(f"\nCột có trong CSV: {list(df.columns)}")
 
 
if __name__ == "__main__":
    main()