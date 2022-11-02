import logging
import pandas as pd
import hydra
import pickle
from omegaconf import DictConfig

from src.data.make_dataset import read_data
from src.entities.predict_pipeline_params import read_predict_pipeline_params, PredictPipelineParams
from src.models.model_fit_predict import predict_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_pipeline(predict_pipeline_params: PredictPipelineParams):
    logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    data = read_data(predict_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    model =  pickle.load(open(predict_pipeline_params.model_path, 'rb'))
    predicts = predict_model(model, data)
    pd.DataFrame(predicts).to_csv(predict_pipeline_params.output_predict_path, index=False)
    logger.info(f"predictions are saved in {predict_pipeline_params.output_predict_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def predict_pipeline_command(cfg: DictConfig):
    params = read_predict_pipeline_params(cfg.predict)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()
