#!/usr/bin/env python
import numpy as np
import pdb
import pandas as pd
import os
from loguru import logger
from src.data.paths import *
import json
from dotenv import find_dotenv, load_dotenv
import json
load_dotenv(find_dotenv())

input_folder = os.path.join(os.getenv('DIR_DATA_RAW'), "data/raw/movies")
output_folder = os.path.join(os.getenv('PROJECT_DIR'), 'temp')
IDS_REV_MISSING = os.path.join(os.getenv('PROJECT_DIR'), 'temp/ids_not_in_rv.json')

not_found_example = ["236251", "236253", "236259", "236265", "236277", "236279", "236291", "236293", "236295", "236305", "236307", "236309", "236315", "236339", "236361", "236379", "236393", "236403", "236463", "236487", "236621", "236623", "236641", "236687", "236709", "236713", "236723", "236875", "236893", "236895", "236935", "236939", "236941", "236943", "236949", "236969", "236975", "236983", "236987", "237005", "237007", "237009", "237015", "237025", "237107", "237127", "237129", "237131", "237187"]
not_found_example = set([int(e) for e in not_found_example])


def read_ids_from_jsonl(fpath):
    ids = []
    with open(fpath, "r") as f:
        for _, line in enumerate(f):
            cont = json.loads(line)
            ids.append(int(cont['item_id']))
    return set(ids)

def read_mainanswers():
    mainanswers = []
    with open(FILE_MAINANSWERS_BOOKS, "r") as reader:
        line = reader.readline()
        while line != "":
            obj = json.loads(line)
            mainanswers.append(obj)
            line = reader.readline()
    mainanswers_df = pd.DataFrame(mainanswers)
    mainanswers_df = mainanswers_df.rename(columns={"user_id":"uid"})
    return mainanswers_df

def save_all_ids():
    # mainanswers.csv
    # metadata.json
    # ratings.json
    # reviews.json
    # tag_count.json
    # tags.json
    ids = {}
    ma = read_mainanswers()
    ma_ids = set([int(e) for e in ma['item_id']])
    mt_ids = read_ids_from_jsonl(os.path.join(input_folder, "metadata.json"))
    rt_ids = read_ids_from_jsonl(os.path.join(input_folder, "ratings.json"))
    rv_ids = read_ids_from_jsonl(os.path.join(input_folder, "reviews.json"))
    tc_ids = read_ids_from_jsonl(os.path.join(input_folder, "tag_count.json"))
    ids["ma"] = list(ma_ids)
    ids["mt"] = list(mt_ids)
    ids["rt"] = list(rt_ids)
    ids["rv"] = list(rv_ids)
    ids["tc"] = list(tc_ids)
    save_path = os.path.join(output_folder, 'ids.json')
    with open(save_path, 'w') as f:
        json.dump(ids, f)
    logger.info(f"Save into {save_path}")
    return ids

def read_ids_json():
    fpath = os.path.join(output_folder, 'ids.json')
    with open(fpath, 'r') as f:
        return json.load(f)

def save_ids_not_in_reviews():
    """ IDs which are not present in reviews"""
    save_path = IDS_REV_MISSING
    ids = read_ids_json()
    diff_ma = set(ids['ma']) - set(ids['rv'])
    diff_mt = set(ids['mt']) - set(ids['rv'])
    diff_rt = set(ids['rt']) - set(ids['rv'])
    diff_tc = set(ids['tc']) - set(ids['rv'])
    ids_missing = diff_ma | diff_mt | diff_rt | diff_tc
    ids_json_obj = {"ids_rv_missing": list(ids_missing)}
    print(f"{len(ids_missing)} IDs are missing in reviews")
    with open(save_path, 'w') as f:
        json.dump(ids_json_obj, f)
    logger.info(f"Save into {save_path}")


def read_missing_ids():
    with open(IDS_REV_MISSING, 'r') as f:
        cont = json.load(f)
        return cont["ids_rv_missing"]


def test_not_found_ids():
    ids = read_ids_json()
    print(ids.keys())
    print(f"in ma: {len(not_found & set(ids['ma']))}")
    print(f"in mt: {len(not_found & set(ids['mt']))}")
    print(f"in rt: {len(not_found & set(ids['rt']))}")
    print(f"in rv: {len(not_found & set(ids['rv']))}")
    print(f"in tc: {len(not_found & set(ids['tc']))}")


def test_ids_integrity():
    ids = read_ids_json()
    diff_ma = set(ids['ma']) - set(ids['rv'])
    diff_mt = set(ids['mt']) - set(ids['rv'])
    diff_rt = set(ids['rt']) - set(ids['rv'])
    diff_tc = set(ids['tc']) - set(ids['rv'])
    print(f"diff ma - rv {len(diff_ma)}")
    print(f"diff mt - rv {len(diff_mt)}")
    print(f"diff rt - rv {len(diff_rt)}")
    print(f"diff tc - rv {len(diff_tc)}")

def run_ids_check():
    save_all_ids()
    save_ids_not_in_reviews()

if __name__ == "__main__":
    if 1 == 0:
        save_all_ids()
    save_ids_not_in_reviews()

