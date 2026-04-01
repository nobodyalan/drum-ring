from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TypeRingConfig:
    model_name: str = "29_TD8_fold5_bs512_lr5e-4_epoch200"
    n_phrase: int = 10
    n_candidate: int = 12
    left_window_mirror: bool = True
    long_tap_threshold: float = 0.4
    min_prediction_interval: float = 0.1
    window_size: int = 20
    stride: int = 2
    stable_count: int = 4


@dataclass(slots=True)
class TypeRingPaths:
    root_dir: Path

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def text_phrases_path(self) -> Path:
        return self.data_dir / "text" / "mackenzie" / "test_phrases.txt"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "model"

    @property
    def char2finger_path(self) -> Path:
        return self.model_dir / "char2finger.json"

    @property
    def corpus_path(self) -> Path:
        return self.model_dir / "language" / "corpus.pkl"

    @property
    def unigrams_path(self) -> Path:
        return self.model_dir / "language" / "unigrams.pkl"

    @property
    def bigrams_path(self) -> Path:
        return self.model_dir / "language" / "bigrams.pkl"

    def touch_detection_model_dir(self, model_name: str) -> Path:
        return self.model_dir / "touch_detection" / model_name

    def validate(self, model_name: str) -> list[str]:
        required = [
            self.text_phrases_path,
            self.char2finger_path,
            self.corpus_path,
            self.unigrams_path,
            self.bigrams_path,
            self.touch_detection_model_dir(model_name) / "args.json",
            self.touch_detection_model_dir(model_name) / "model.pt",
        ]
        return [str(path) for path in required if not path.exists()]
