from dataclasses import dataclass
from src.entities.split_params import SplittingParams
from src.entities.feature_params import FeatureParams
from src.entities.train_params import TrainingParams
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(cfg: DictConfig) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    return schema.load(cfg)
