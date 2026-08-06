import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path


class PipelineConfig(BaseModel):
    device: str = "cuda"
    precision: str = "fp32"
    compile_models: bool = False
    self_ensemble: bool = False
    lowlight_denoise: bool = False  # When False, skip Restormer denoising stage


class RouterConfig(BaseModel):
    haze_threshold: float = 0.45
    lowlight_threshold: float = 0.50
    rain_threshold: float = 0.45
    execution_order: List[str] = ["lowlight", "rain", "haze"]
    detector_type: str = "learned"  # "heuristic" or "learned"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/restoration.log"


class SystemConfig(BaseModel):
    pipeline: PipelineConfig
    router: RouterConfig
    logging: LoggingConfig


class ModelParams(BaseModel):
    name: str
    checkpoint_path: str
    download_url: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelsRegistryConfig(BaseModel):
    dehazeformer: ModelParams
    retinexformer: ModelParams
    restormer: ModelParams
    restormer_denoise: ModelParams


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_system_config(config_path: str = "configs/config.yaml") -> SystemConfig:
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        data = yaml.safe_load(f)
    config = SystemConfig(**data)
    config.logging.file = str(PROJECT_ROOT / config.logging.file)
    return config


def load_models_config(
    config_path: str = "configs/models.yaml",
) -> ModelsRegistryConfig:
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        data = yaml.safe_load(f)
    config = ModelsRegistryConfig(**data)
    # Resolve all checkpoint paths to absolute paths
    for model_cfg in [config.dehazeformer, config.retinexformer, config.restormer, config.restormer_denoise]:
        model_cfg.checkpoint_path = str(PROJECT_ROOT / model_cfg.checkpoint_path)
    return config


# Global configuration singletons
SYSTEM_CONFIG = None
MODELS_CONFIG = None


def get_config() -> SystemConfig:
    global SYSTEM_CONFIG
    if SYSTEM_CONFIG is None:
        SYSTEM_CONFIG = load_system_config()
    return SYSTEM_CONFIG


def get_models_config() -> ModelsRegistryConfig:
    global MODELS_CONFIG
    if MODELS_CONFIG is None:
        MODELS_CONFIG = load_models_config()
    return MODELS_CONFIG
