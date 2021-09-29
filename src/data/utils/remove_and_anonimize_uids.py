#!/usr/bin/env python
from loguru import logger
import pandas as pd
import json
import os
import numpy as np
from src.data.paths import PROJECT_DIR


file_exclude_ids = "/home/ms314/projects/tagnav-code-refactored/data/raw/exlcude_usersIds.csv"
input_folder = "/home/ms314/projects/tagnav-code-refactored/data/raw/movies"
output_folder = "/home/ms314/projects/tagnav-code-refactored/data/raw/movies_cleaned_ids"
file_interim_ratings = os.path.join(output_folder, "_interim_ratings_excluded_ids.json")


def anonim_survey():
    input_file = os.path.join(input_folder, 'mainanswers.csv')
    output_file = os.path.join(output_folder, 'mainanswers.csv')
    an = IdsAnonimizer()
    mappings = an.read_anonim_mappings()
    df = pd.read_csv(input_file)
    df['uid'] = df['uid'].apply(lambda x: mappings[str(x)])
    df.to_csv(output_file, index=False)
    logger.info(f"Dump into {output_file}")


def anonim_ratings():
    input_file = file_interim_ratings
    output_file = os.path.join(output_folder, 'ratings.json')
    an = IdsAnonimizer()
    mappings = an.read_anonim_mappings()
    with open(input_file, 'r') as f, open(output_file, 'w') as fout:
        for line in f:
            cont = json.loads(line)
            old_id = cont['user_id']
            new_id = mappings[str(old_id)]
            cont['user_id'] = new_id
            fout.write(json.dumps(cont)+'\n')
    logger.info(f"Dump into {output_file}")


class IdsAnonimizer:
    mappings_file = os.path.join(PROJECT_DIR, 'temp/ids_random_mapping.json')

    @staticmethod
    def read_anonim_mappings():
        with open(IdsAnonimizer.mappings_file, 'r') as f:
            cont = json.load(f)
            return cont

    @staticmethod
    def create_anonim_mappings():
        ids = IdsAnonimizer.get_all_users_ids()
        n = len(ids)
        rand_ids = np.random.choice(max(n, 10**6), n)
        mapping = {}
        for k, id in enumerate(ids):
            mapping[str(id)] = int(rand_ids[k])
        with open(IdsAnonimizer.mappings_file, 'w') as f:
            json.dump(mapping, f)
        logger.info(f"Dump into {IdsAnonimizer.mappings_file}")


    @staticmethod
    def get_all_users_ids():
        ids_r = IdsAnonimizer.ids_from_ratings()
        ids_m = IdsAnonimizer.ids_from_mainanswers()
        return ids_r | ids_m

    @staticmethod
    def ids_from_mainanswers():
        input_file = os.path.join(input_folder, "mainanswers_orig_uids.csv")
        df = pd.read_csv(input_file)
        ids = df['uid'].to_numpy()
        ids = set(ids)
        return ids

    @staticmethod
    def ids_from_ratings():
        # file with already removed ids from exclude list
        input_file = os.path.join(output_folder, "ratings_excluded_ids.json")
        uids = []
        with open(input_file, 'r') as f:
            for line in f:
                cont = json.loads(line)
                uids.append(cont['user_id'])
        uids = set(uids)
        return uids


def exclude_uids():
    df = pd.read_csv(file_exclude_ids, header=None).to_numpy().tolist()
    df = [e[0] for e in df]
    df.extend([173778, 51714, 166954, 114092, 141799, 166342])
    return df


def remove_uids_from_ratings():
    input_file = os.path.join(input_folder, "ratings_orig_uids.json")
    output_file = file_interim_ratings
    exclude = exclude_uids()
    users_ids_before = set()
    users_ids_after = set()
    item_ids_before = set()
    item_ids_after = set()
    with open(input_file, 'r') as f, open(output_file, 'w') as fout:
        count_excluded = 0
        tot = 0
        count_written = 0
        for line in f:
            tot += 1
            cont = json.loads(line)
            users_ids_before.add(cont['user_id'])
            item_ids_before.add(cont['item_id'])
            if cont['user_id'] in exclude:
                count_excluded += 1
                continue
            fout.write(line)
            count_written += 1
            users_ids_after.add(cont['user_id'])
            item_ids_after.add(cont['item_id'])
    print(f"Excluded {count_excluded} raws out of {tot} = {100.0*count_excluded/float(tot)} %")
    print(f"UIDs before = {len(users_ids_before)}")
    print(f"UIDs after = {len(users_ids_after)}")
    print(f"Items IDs before = {len(item_ids_before)}")
    print(f"Items IDs after = {len(item_ids_after)}")
    print(f"Rows number before = {tot}")
    print(f"Rows number after = {count_written}")



if __name__ == "__main__":
    remove_uids_from_ratings()
    an = IdsAnonimizer()
    an.create_anonim_mappings()
    anonim_ratings()
    anonim_survey()

