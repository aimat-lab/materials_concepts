import re
import os
import csv
import tabulate

STEP_SIZE = 2000

regex = re.compile(r"(\d+)_(\d+)")
inf_dir = "/pfs/work7/workspace/scratch/{CLUSTER_USER}-matconcepts/data/inference_13B/full-finetune-100/"


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_line_count(path):
    with open(path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)
        return sum(1 for _ in csv_reader)


data = [["File", "Ok", "Line Count"]]

for f in os.listdir(inf_dir):
    plain_name = get_filename(inf_dir + f)
    print(plain_name)
    line_count = get_line_count(inf_dir + f)
    mo = regex.search(plain_name)
    formatted_name = f"{mo.group(1)}-{mo.group(2)}"
    data.append([formatted_name, "✓" if line_count == STEP_SIZE else "✗", line_count])

print(tabulate.tabulate(data, headers="firstrow"))
