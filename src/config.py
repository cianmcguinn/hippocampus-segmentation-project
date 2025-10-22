# src/config.py
from dataclasses import dataclass

@dataclass
class Config:
    SEED: int = 42
    ENCODER_NAME: str = "timm-mobilenetv3_small_075"  # baseline
    ENCODER_WEIGHTS: str = "imagenet"
    IN_CHANNELS: int = 1
    NUM_CLASSES: int = 3

CFG = Config()
