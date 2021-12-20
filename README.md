# Revisiting the Tag Relevance Prediction Problem

## Introduction

This repository contains data pipeline to run experiments described in [Vig et al., 2012] and [Kotkov et al., 2021]. The pipeline takes item and tag data as input and produces scores that indicate degrees, to which tags apply to items. The pipeline consists of the following steps:
* Raw data transformation – converting raw data to a suitable format with `./data/make_interim_pickle_files.py`
* Feature generation – extracting features from the converted data with `./features/build_features.py` and `./models/r/build_features.R`
* Data split – preparing data for evaluation with `./data/make_10folds.py`
* Prediction – predicting scores based on extracted features with `./models/r/run_tenfolds.R` and `./models/run_tenfolds.py`
* Evaluation – measuring performance of algorithms with `./models/calcl_mae_for_folds.py`

## Datasets

* The Tag Genome dataset for movies: https://grouplens.org/datasets/movielens/tag-genome-2021/
* The Tag Genome dataset for book: https://grouplens.org/datasets/book-genome

## Usage License

This work is licensed under the Creative Commons Attribution-NonCommercial 3.0 License. If you are using these code and/or datasets, please cite the following papers:

- [[Kotkov et al., 2021] Kotkov, D., Maslov, A., and Neovius, M. (2021). Revisiting the tag relevance prediction problem. In Proceedings of the 44th International ACM SIGIR conference on Research and Development in Information Retrieval.](https://doi.org/10.1145/3404835.3463019)
- [[Vig et al., 2012] Vig, J., Sen, S., and Riedl, J. (2012). The tag genome: Encoding community knowledge to support novel interaction. ACM Trans. Interact. Intell. Syst., 2(3):13:1–13:44.](https://dl.acm.org/doi/abs/10.1145/2362394.2362395)

==============================

## Set up

```
python -m venv venv
source venv/bin/activate
pip install -e .
```

Create `.env` file with the path to the project directory

```
# Environment variables go here, can be read by `python-dotenv` package:
#
#   `src/script.py`
#   ----------------------------------------------------------------
#    import dotenv
#
#    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
#    dotenv_path = os.path.join(project_dir, '.env')
#    dotenv.load_dotenv(dotenv_path)
#   ----------------------------------------------------------------
#
# DO NOT ADD THIS FILE TO VERSION CONTROL!
PROJECT_DIR="/home/u/project_dir"
DIR_DATA_RAW="/home/u/../data/raw"
```

## Run 

1) `src/data/make_interim_pickle_files.py`
2) `src/features/build_features.py`
3) Run models (R/PyTorch)

or, instead 1) and 2)

```
cd src/
chmod +x run.sh 
./run.sh
```

## Input data format 

```
data/raw_example
```


## Data flow

```
mainanswers.txt -> build_features -> Input files for R -> dump for r and Pytorch models
```

## Generating Tag Genome scores
To generate Tag Genome scores, run the following scripts:
1. `src/run.sh`
2. `src/models/r/generate_features_for_scores.R`
3. `src/models/r/generate_scores.R`
4. `src/models/generate_scores.py`

Scripts:
* `src/models/r/generate_features_for_scores.R` – generates features for prediction algorithms. The script requires the file `data/processed/movie_ids.csv`, which contains ids of items to included in the Tag Genome scores. The file contains ids in the csv format in the field: `movie_id`.
* `src/models/r/generate_scores.R` – generates scores with the method presented in [Vig et al., 2012]
* `src/models/generate_scores.py` – generates scores with TagDL [Kotkov et al., 2021]


## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── predictions    <- Preditions of models.
    │   ├── pickle_files   <- Serrialized objects.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The feature data sets for modeling.
    │   ├── raw            <- The original, immutable data dump (item data).
    │   └── raw_example    <- Example of the raw data
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── run.sh         <- The script for running the project
    │   │
    │   ├── data           <- Scripts to generate intermediate data
    │   │   └── make_interim_pickle_files.py
    │   │
    │   ├── features       <- Scripts to turn intermediate data into features for modeling
    │   │   └── build_features.py
    │   │
    │   └── models         <- Scripts to train models and make predictions (+ logistic regression for calculating the tag_prob feature)
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


## Acknowledgements
We would like to thank GroupLens for providing us with the dataset and code for the regression algorithm [Vig et al., 2012]. We would also like to thank organizations that supported publication of this dataset: the Academy of Finland, grant #309495 (the LibDat project) and the Academy of Finland Flagship programme: Finnish Center for Artificial Intelligence FCAI.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
