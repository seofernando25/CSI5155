from models import MODEL_LABELS
from sampling import SAMPLING_LABELS

FIGURE_CONFIG = {
    "dpi": 300,
    "bbox_inches": "tight",
    "facecolor": "white",
    "edgecolor": "none",
}

FONT_CONFIG = {
    "title": 16,
    "subtitle": 14,
    "label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}

FIGURE_SIZES = {
    "large": (18, 12),
    "medium": (15, 10),
    "small": (12, 8),
    "wide": (20, 8),
    "tall": (10, 12),
}

HISTOGRAM_CONFIG = {"bins": 30, "alpha": 0.7, "edgecolor": "black"}

# Labels
MODEL_LABELS = MODEL_LABELS
SAMPLINGS = SAMPLING_LABELS
