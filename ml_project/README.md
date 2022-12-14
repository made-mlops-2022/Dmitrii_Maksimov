ml_project
==============================

1_Howerok on MLOPS at the MADE

Setup: 
~~~
make create_environment
make requirements
~~~

Run train: 
~~~
python -m src.train_pipeline train=<config_name> <changes>
example: 
python -m src.train_pipeline train=train_config_svm train splitting_params.val_size=0.2
~~~

Run predict: 
~~~
python -m src.predict_pipeline predict=<config_name> <changes>
example: 
python -m src.predict_pipeline predict.model_path=models/model_svm.pkl
~~~

Run test: 
~~~
make test
~~~

Run lint: 
~~~
make lint
~~~

Run coverage: 
~~~
make coverage
~~~

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── configs            <- Configs for training and predictions
    │
    ├── data
    │   ├── predict        <- data for prediction
    │   └── raw            <- The original, immutable data dump.
    │
    ├── predict            <- Model predictions
    │
    ├── metrics            <- Model summaries
    │
    ├── models             <- Trained and serialized models
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── entities           <- Parameters
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to get/save models
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │    └── visualize.py
    │   ├── entities           <- Parameters
    │   │
    │   ├── train_pipeline.py, predict_pipeline.py           <- Scripts for prediction and training
