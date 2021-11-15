#!/usr/bin/env python
import pandas as pd
from src.common import get_path
import math

from src.models.parameters import NUM_EPOCHS
import run_tenfolds

TRAINING_DATA_PATH = get_path("data/processed/features_r.csv")
FEATURES_PATH = get_path("data/processed/features_for_score_generation.csv")
MODEL_TO_SAVE_PATH = get_path("data/processed/model.bin")
PERFORMANCE_REPORT_PATH = get_path('data/processed/perf.json')
TAG_GENOME_PATH = get_path('data/processed/tagdl.csv')


train_data = pd.read_csv(TRAINING_DATA_PATH)
test_df = pd.read_csv(FEATURES_PATH)
test_df["targets"] = 0

tags = set()
tags = tags.union(test_df.tag.unique())
tags = tags.union(train_data.tag.unique())

train_loader, validation_loader = run_tenfolds.create_train_validation_dataloader(train_data)

run_tenfolds.train(train_loader,
                   validation_loader=validation_loader,
                   num_epochs=NUM_EPOCHS,
                   model_path_save=MODEL_TO_SAVE_PATH,
                   perf_json_save_path=PERFORMANCE_REPORT_PATH)

# we split the dataframe into parts, because otherwise the server kills the process
part_size = 10
model = run_tenfolds.get_model(MODEL_TO_SAVE_PATH)
predictions = pd.DataFrame()
for i in range(math.ceil(len(test_df) / part_size)):
    start = i * part_size
    end = (i + 1) * part_size
    test_part_copy = test_df.iloc[start:end].copy()
    test_part_copy.index = list(range(len(test_part_copy)))
    test_loader = run_tenfolds.create_dataloader(test_part_copy)
    predictions_part = run_tenfolds.predict(test_loader, model)
    predictions = pd.concat([predictions, predictions_part])

predictions.index = list(range(len(predictions)))
test_df.index = list(range(len(test_df)))
test_df["score"] = predictions[0]
test_df["score"] = (test_df["score"] - 1) / 4

test_df[["tag", "movieId", "score"]].to_csv(TAG_GENOME_PATH, header=None, index=False)