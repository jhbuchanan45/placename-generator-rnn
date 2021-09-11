import numpy as np
from tqdm import tqdm
import os
import csv
from pathlib import Path
import unicodedata
import string

DATA_PATH = "raw/data"

all_letters = string.ascii_letters + " -'"
fieldnames = open("./raw/data_header.csv").read().split(',')
region_name_set = set()
names_by_region = dict()


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def check_row(row):
    name = unicode_to_ascii(row['NAME1'])

    if name[0].isupper() and (name[1].isupper() or name[1].isspace()):
        raise Exception("dodgy looking", name)


for filename in tqdm(os.listdir(DATA_PATH)):
    path = os.path.join(DATA_PATH, filename)

    with open(path, mode="r") as csv_data:
        reader = csv.DictReader(csv_data, fieldnames=fieldnames)

        for row in reader:
            try:
                check_row(row)
                region_name_set.add(unicode_to_ascii(row['REGION']))
                names_by_region.setdefault(unicode_to_ascii(row['REGION']), []).append(
                    [unicode_to_ascii(row['NAME1']), unicode_to_ascii(row['COUNTY_UNITARY'])])
            except Exception as e:
                print(e)
                pass

print(region_name_set)
Path("processed").mkdir(exist_ok=True)
np.save("processed/region_names.npy", list(region_name_set))

# save by region
totalnames = 0
for name in region_name_set:
    names_arr = names_by_region[name]
    np.random.shuffle(names_arr)
    Path(f"processed/{name}").mkdir(exist_ok=True)

    VAL_PCT = 0.1
    val_size = int(len(names_arr) * VAL_PCT)

    train_names = names_arr[:-val_size]
    test_names = names_arr[-val_size:]

    np.save(f"processed/{name}/placenames.train.npy", train_names)
    np.save(f"processed/{name}/placenames.test.npy", test_names)
    print(f"\n{name}: {len(names_arr)} entries; (Test: {len(test_names)}, Train: {len(train_names)})")
    totalnames += len(names_arr)
