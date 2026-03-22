# ── App-wide configuration ───────────────────────────────────────────────────

MODEL_PATH         = "cifar100_modelv2.h5"
INPUT_SIZE         = (32, 32)          # model input resolution
CONFIDENCE_THRESHOLD = 0.40           # below this → "unrecognized"
TOP_K              = 5                # how many predictions to show
SUPPORTED_FORMATS  = ["jpg", "jpeg", "png", "bmp", "webp"]

# Gradient stops for top-5 bar colours (rank 0 → 4)
BAR_COLOURS = ["#7c6af0", "#9d8ef5", "#b8aaf7", "#ccc4f9", "#dddafb"]