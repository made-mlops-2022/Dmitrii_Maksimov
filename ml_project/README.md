ml_project
==============================

1_Howerok on MLOPS at the MADE

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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
