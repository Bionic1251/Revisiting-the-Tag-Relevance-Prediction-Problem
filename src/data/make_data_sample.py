#!/usr/bin/env python
"""
    Make input raw data sample for testing purposes
"""
import numpy as np
import pandas as pd
import os
import shutil
from loguru import logger
from pathlib import Path
import json
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

input_folder = os.path.join(os.getenv('PROJECT_DIR'), "data/raw/movies")
output_folder = os.path.join(os.getenv('PROJECT_DIR'), "data/raw/movies_sample")


def create_foldere_if_not_exists(folder_path):
    try:
        Path(folder_path).mkdir(parents=True, exist_ok=False)
        logger.info(f"Created folder {folder_path}")
    except FileExistsError:
        pass

def read_ids_from_jsonl(fpath):
    ids = []
    with open(fpath, "r") as f:
        for k, line in enumerate(f):
            cont = json.loads(line)
            ids.append(int(cont['item_id']))
    return set(ids)


def filter_jsonl_file(fname, ids):
    inpath = os.path.join(input_folder, fname)
    outpath = os.path.join(output_folder, fname)
    count = 0
    with open(inpath, "r") as f, open(outpath, "w") as fout:
        for k, line in enumerate(f):
            cont = json.loads(line)
            if int(cont['item_id']) in ids:
                fout.write(json.dumps(cont) + "\n")
                count += 1
    logger.info(f"Copy {count} lines from {inpath} to {outpath}")



def read_ids():
    ma = pd.read_csv(os.path.join(input_folder, "mainanswers.csv"))
    ma_ids = set([int(e) for e in ma['item_id']])
    mt_ids = read_ids_from_jsonl(os.path.join(input_folder, "metadata.json"))
    rt_ids = read_ids_from_jsonl(os.path.join(input_folder, "ratings.json"))
    rv_ids = read_ids_from_jsonl(os.path.join(input_folder, "reviews.json"))
    tc_ids = read_ids_from_jsonl(os.path.join(input_folder, "tag_count.json"))
    all_ids = ma_ids & mt_ids & rt_ids & rv_ids & tc_ids
    #logger.info(f"208 in all_ids {208 in all_ids}")
    return all_ids


def make_sample(size:int):
    shutil.copyfile(os.path.join(input_folder, "tags.json"), os.path.join(output_folder, "tags.json"))
    all_ids = read_ids()
    logger.info(f"IDs sets intersection size from all files {len(all_ids)}")
    all_ids = set(list(all_ids)[:size])
    filter_jsonl_file("metadata.json", all_ids)
    filter_jsonl_file("ratings.json", all_ids)
    filter_jsonl_file("reviews.json", all_ids)
    filter_jsonl_file("tag_count.json", all_ids)
    ma = pd.read_csv(os.path.join(input_folder, "mainanswers.csv"))
    logger.info(f"Mainanswers shape before: {ma.shape}")
    ma = ma[ma['item_id'].isin(all_ids)]
    logger.info(f"Mainanswers after: {ma.shape}")
    ma.to_csv(os.path.join(output_folder, "mainanswers.csv"), index=False)


# Input files
#mainanswers.csv
#metadata.json
#ratings.json
#reviews.json
#tag_count.json
#tags.json

if __name__ == "__main__":
    create_foldere_if_not_exists(output_folder)
    make_sample(size=100)
