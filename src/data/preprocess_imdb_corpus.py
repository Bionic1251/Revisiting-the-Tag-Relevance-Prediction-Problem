"""
Parse and clean imdbcorpus.json into csv file in order to load into MySQL.
"""

from loguru import logger
import tempfile
import csv
import os
import json
import re


#del re_curly_braces_not_json = re.compile(r'{[^"][^{]*[^"]}')
#del re_letter_and_plus_in_curly_braces = re.compile(r'{\w+\+}')
re_del_comma_add_new_line = re.compile(r'"\}.{0,5}\{"')


def apply_re_patterns(x):
    x_proc = x
    x_proc = re_del_comma_add_new_line.sub('"}\n{"', x_proc)
    return x_proc


def json_to_jsonl(input_file, output_file=None, buffer_size=2):

    def pre_proc(ch):
        ch_str = ch.decode("utf-8")
        ch_str = ch_str.replace('[', '')
        ch_str = ch_str.replace(']', '')
        ch_str = apply_re_patterns(ch_str)
        return ch_str

    f_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='./temp')
    print(f"Temp file: {f_temp.name}")
    with open(input_file, 'rb') as f_in:
        chunk = f_in.read(buffer_size)
        chunk_str = pre_proc(chunk)
        f_temp.write(chunk_str)
        while chunk:
            chunk = f_in.read(buffer_size)
            chunk_str = pre_proc(chunk)
            f_temp.write(chunk_str)
    f_temp.close()
    with open(output_file, 'w') as f_out, open(f_temp.name, 'r') as fp:
        for line in fp:
            f_out.write(apply_re_patterns(line))
    f_temp.close()
    logger.info(f"Save into {output_file}")


def jsonl_to_csv(input_file, out_file, errors_file):
    count_errors = 0
    count_lines = 0
    with open(input_file, 'r') as f, \
            open(out_file, 'w') as f_csv, open(errors_file, 'w') as errf:
        csv_writer = csv.writer(f_csv, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i, line in enumerate(f):
            try:
                cont = json.loads(line)
            except Exception as e:
                count_errors += 1
                errf.write(line)
                errf.write("\n---\n")
            movie_id = cont['movieId']
            heading = cont['heading'].replace('\t', ' ').replace(r'\r', '')
            body = cont['body'].replace('\t', ' ').replace(r'\r', '')
            heading = heading.replace('\t', ' ').replace(r'\r', '')
            body = body.replace('\t', ' ')
            csv_writer.writerow([movie_id, heading, body])
            count_lines += 1
    logger.info(f"Write into {out_file}")
    print(f"count_errors={count_errors}")
    print(f"count lines={count_lines}")


def p(x):
    main_path = "/home/ms314/datasets/tagnav/json_data/"
    return os.path.join(main_path, x)


if __name__ == "__main__":
    json_to_jsonl(p("imdbcorpus.json"), p("imdbcorpus_out.json"), buffer_size=1000000)
    jsonl_to_csv(p("imdbcorpus_out.json"), p("out_csv/imdbcorpus_out.csv"), p("errors.txt"))
