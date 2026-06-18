"""
config.py — Cấu hình trung tâm cho đồ án ICF Detector
=======================================================
Tất cả các file khác đều import từ đây.
KHÔNG hardcode API key, path, hay hyperparameter ở chỗ khác.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
# Tìm .env từ thư mục gốc project (một cấp trên src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ══════════════════════════════════════════════════════════════════════════════
# 1. API
# ══════════════════════════════════════════════════════════════════════════════

YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")

# Số kết quả tối đa khi search kênh mới (dùng trong expand.py)
API_SEARCH_MAX_RESULTS: int = 50

# Số video gần nhất lấy per channel để tính features
# (50 là giới hạn 1 request của YouTube API — đủ cho time-series)
VIDEOS_PER_CHANNEL: int = 50


# ══════════════════════════════════════════════════════════════════════════════
# 2. Đường dẫn (Paths)
# ══════════════════════════════════════════════════════════════════════════════

class Paths:
    ROOT        = _PROJECT_ROOT
    SRC         = ROOT / "src"
    DATA        = ROOT / "data"
    NOTEBOOKS   = ROOT / "notebooks"
    MODELS      = ROOT / "models"

    # Data sub-folders
    RAW         = DATA / "raw"
    COLLECTED   = DATA / "collected"
    PROCESSED   = DATA / "processed"

    # Seed files (2 file bạn đã có)
    SEED_SLOP   = RAW / "AI slop.txt"
    SEED_GENUINE= RAW / "non AI.txt"

    # Output của crawl.py
    CHANNELS_RAW = COLLECTED / "channels_raw.csv"

    # Output của features.py (input của model)
    FEATURES_FINAL = PROCESSED / "features_final.csv"

    # Saved models
    MODEL_RF    = MODELS / "random_forest.pkl"
    MODEL_IF    = MODELS / "isolation_forest.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Labels
# ══════════════════════════════════════════════════════════════════════════════

LABEL_SLOP    = 1   # Inauthentic channel
LABEL_GENUINE = 0   # Authentic channel


# ══════════════════════════════════════════════════════════════════════════════
# 4. Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

# Feature 1 — Time series
# Số ngày tối thiểu giữa 2 lần upload để không bị coi là "burst"
MIN_UPLOAD_INTERVAL_DAYS: float = 0.5

# Feature 6 — Text similarity
# Số video gần nhất dùng để tính avg channel similarity
SIMILARITY_WINDOW: int = 10

# Ngưỡng similarity để coi là "highly repetitive" (dùng trong EDA)
SIMILARITY_HIGH_THRESHOLD: float = 0.8

# TF-IDF max features (dùng trong features.py)
TFIDF_MAX_FEATURES: int = 5000


# ══════════════════════════════════════════════════════════════════════════════
# 5. Anomaly Detection (anomaly.py — Isolation Forest)
# ══════════════════════════════════════════════════════════════════════════════

IF_CONTAMINATION: float = 0.15   # Ước tính ~15% data là anomaly
IF_N_ESTIMATORS : int   = 100
IF_RANDOM_STATE : int   = 42


# ══════════════════════════════════════════════════════════════════════════════
# 6. Model (train.py — Random Forest)
# ══════════════════════════════════════════════════════════════════════════════

RF_N_ESTIMATORS : int   = 200
RF_MAX_DEPTH    : int   = 8       # Giới hạn để tránh overfit với dataset nhỏ
RF_RANDOM_STATE : int   = 42
RF_TEST_SIZE    : float = 0.2     # 80/20 train/test split
RF_CV_FOLDS     : int   = 5       # Số fold cross-validation


# ══════════════════════════════════════════════════════════════════════════════
# 7. Seed Expansion (expand.py)
# ══════════════════════════════════════════════════════════════════════════════

# Kênh nằm trong vùng ambiguous (gần cả 2 centroid) → cần label thủ công
EXPANSION_AMBIGUOUS_THRESHOLD: float = 0.3   # khoảng cách tương đối

# Số kênh candidate tối đa mỗi lần expand
EXPANSION_MAX_CANDIDATES: int = 200


# ══════════════════════════════════════════════════════════════════════════════
# 8. Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate() -> None:
    """
    Kiểm tra config hợp lệ trước khi chạy pipeline.
    Gọi hàm này ở đầu main.py hoặc crawl.py.
    """
    errors = []

    if not YOUTUBE_API_KEY:
        errors.append(
            "YOUTUBE_API_KEY chưa được set.\n"
            "  → Tạo file .env từ .env.example rồi điền key vào."
        )

    if not Paths.SEED_SLOP.exists():
        errors.append(f"Không tìm thấy seed file: {Paths.SEED_SLOP}")

    if not Paths.SEED_GENUINE.exists():
        errors.append(f"Không tìm thấy seed file: {Paths.SEED_GENUINE}")

    # Tự tạo thư mục output nếu chưa có
    for folder in [Paths.COLLECTED, Paths.PROCESSED, Paths.MODELS]:
        folder.mkdir(parents=True, exist_ok=True)

    if errors:
        raise EnvironmentError(
            "\n\n[config] Lỗi cấu hình:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    print("[config] ✓ Cấu hình hợp lệ.")
    print(f"  PROJECT_ROOT : {Paths.ROOT}")
    print(f"  API key      : {YOUTUBE_API_KEY[:8]}{'*' * (len(YOUTUBE_API_KEY) - 8)}")
    print(f"  Seed slop    : {Paths.SEED_SLOP.name} ({_count_lines(Paths.SEED_SLOP)} kênh)")
    print(f"  Seed genuine : {Paths.SEED_GENUINE.name} ({_count_lines(Paths.SEED_GENUINE)} kênh)")


def _count_lines(path: Path) -> int:
    """Đếm số dòng không rỗng trong file txt."""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# ── Quick self-test khi chạy trực tiếp ────────────────────────────────────────
if __name__ == "__main__":
    validate()