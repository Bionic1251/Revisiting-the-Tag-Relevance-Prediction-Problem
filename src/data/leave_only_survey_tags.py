"""
Delete tags from tag_count and from tags. json which are not in mainanwers
"""
import json
import os
import pandas as pd
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


IN_DIR = os.path.join(os.getenv('PROJECT_DIR'), "data/raw/movies")
OUT_DIR = os.path.join(os.getenv('PROJECT_DIR'), "data/raw/movies_cleaned_tags")

def filter_tags(input_file, output_file, tags_ids_to_leave):
    count_in = 0
    count_out = 0
    with open(input_file, "r") as f, open(output_file, 'w') as fout:
        for _, line in enumerate(f):
            count_in += 1
            cont = json.loads(line)
            try:
                tag_id = int(cont['id'])
            except KeyError:
                try:
                    tag_id = int(cont['tag_id'])
                except KeyError as e:
                    raise(e)

            if tag_id in tags_ids_to_leave:
                fout.write(json.dumps(cont) + "\n")
                count_out += 1
    print(count_in, count_out)



if __name__ == "__main__":
    ma = pd.read_csv(os.path.join(IN_DIR, "mainanswers.csv"))
    ma_ids = set([int(e) for e in ma['tag_id']])
    filter_tags(os.path.join(IN_DIR, "tags.json"), os.path.join(OUT_DIR, "tags.json"), ma_ids)
    filter_tags(os.path.join(IN_DIR, "tag_count.json"), os.path.join(OUT_DIR, "tag_count.json"), ma_ids)
