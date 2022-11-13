from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    model_path: str
    output_predict_path: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(cfg: DictConfig) -> PredictPipelineParams:
    schema = PredictPipelineParamsSchema()
    return schema.load(cfg)
