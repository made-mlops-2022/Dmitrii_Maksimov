import json
import logging
import hydra
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline

from src.data.make_dataset import read_data, split_train_val_data
from src.entities.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params
from src.features.build_features import extract_target, build_transformer
from src.models.model_fit_predict import get_model, serialize_model, predict_model, evaluate_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(data, training_pipeline_params.splitting_params)
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    model = Pipeline([
        ("transformer", transformer),
        ("model", get_model(training_pipeline_params.train_params))
    ])
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    model.fit(train_df, train_target)

    val_target = extract_target(val_df, training_pipeline_params.feature_params)

    predicts = predict_model(model, val_df)

    metrics = evaluate_model(predicts, val_target)

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    serialize_model(model, training_pipeline_params.output_model_path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_pipeline_command(cfg: DictConfig):
    params = read_training_pipeline_params(cfg.train)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
