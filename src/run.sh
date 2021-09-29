#!/bin/bash 

rm -r ../data/pickle_files/*
touch ../data/pickle_files/.gitkeep

rm -r ../data/interim/*
touch ../data/interim/.gitkeep

rm -r ../data/processed/*
touch ../data/processed/.gitkeep

rm -r ../data/predictions/*
touch ../data/predictions/.gitkeep

mkdir ../data/processed/10folds/
touch ../data/processed/10folds/.gitkeep

# Python build features
./data/make_interim_pickle_files.py
./features/build_features.py

# R build features
cd models/r
Rscript build_features.R
#- Only if needed: Rscript make_predictions.R

# Split data set with features into 10 folds
cd ../../
./data/make_10folds.py
 
# R model run for 10 folds
cd models/r
Rscript run_tenfolds.R
 
# PyTorch model run for 10 folds
cd ../../
./models/run_tenfolds.py
 
# Calculate and compare performance for PyTorch and R glmer models
./models/calcl_mae_for_folds.py
