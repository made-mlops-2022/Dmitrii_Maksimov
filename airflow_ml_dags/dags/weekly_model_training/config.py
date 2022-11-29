from dataclasses import dataclass, field
from typing import List, Optional
from marshmallow_dataclass import class_schema
from omegaconf import OmegaConf


@dataclass()
class SplittingParams:
    train_path: str
    val_path: str
    val_size: float = field(default=0.2)
    random_state: int = field(default=123)


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    features_to_drop: List[str]
    target_col: Optional[str]


@dataclass()
class TrainingParams:
    model_type: str = field(default="SupportVectorMachine")
    random_state: int = field(default=123)


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


def read_training_pipeline_params(cfg_path: str) -> TrainingPipelineParams:
    cfg = OmegaConf.load(cfg_path)
    TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
    schema = TrainingPipelineParamsSchema()
    return schema.load(cfg)


def replace_ds_nodas(date: str, path: str) -> str:
    return date.join(path.split('{{ ds_nodash }}'))
