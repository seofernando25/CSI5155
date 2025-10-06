from typing import Literal, Optional
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

SamplingStrategy = Literal["none", "under", "smote"]

SAMPLING_STRATEGIES: tuple[SamplingStrategy, ...] = ("none", "under", "smote")

SAMPLING_LABELS: dict[SamplingStrategy, str] = {
    "none": "No Sampling",
    "under": "Undersampling",
    "smote": "SMOTE",
}


def get_sampler(sampling: Optional[SamplingStrategy]):
    if sampling is None or sampling == "none":
        return None
    if sampling == "smote":
        return SMOTE()
    if sampling == "under":
        return RandomUnderSampler()


