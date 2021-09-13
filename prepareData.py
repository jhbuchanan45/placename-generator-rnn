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
names_by_length = list()


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def check_row(row):
    name = unicode_to_ascii(row['NAME1'])

    if len(name) < 2 or (name[0].isupper() and (name[1].isupper() or name[1].isspace())):
        return False
    return True


def safe_list(main_list, index):
    if index >= len(main_list):
        main_list.extend([None] * (index + 1 - len(main_list)))

    if not isinstance(main_list[index], list):
        main_list[index] = list()

    # print(len(main_list))
    # print(index)

    return main_list[index]


total_rows = 0
for filename in tqdm(os.listdir(DATA_PATH)):
    path = os.path.join(DATA_PATH, filename)

    with open(path, mode="r") as csv_data:
        reader = csv.DictReader(csv_data, fieldnames=fieldnames)

        for row in reader:
            try:
                if check_row(row):
                    total_rows += 1
                    region = unicode_to_ascii(row['REGION'])
                    region_name_set.add(region)
                    name = unicode_to_ascii(row['NAME1'])
                    word_list = safe_list(names_by_length, len(name))
                    word_list.append((name, region))
            except Exception as e:
                print("error: ", e)
                pass

print(total_rows)
print(region_name_set)

test_names_by_length = [[]] * len(names_by_length)
train_names_by_length = [[]] * len(names_by_length)

print(len(names_by_length))

total_words = 0

for word_len, words in enumerate(names_by_length):
    if words is None:
        words = list()

    np.random.shuffle(words)
    # print(words)
    total_words += len(words)

    VAL_PCT = 0.05
    val_size = int(len(words) * VAL_PCT)

    train_names_by_length[word_len] = words[:-val_size]
    test_names_by_length[word_len] = words[-val_size:]

print(total_words)

Path("processed").mkdir(exist_ok=True)
np.save("processed/region_names.npy", list(region_name_set))
np.save("processed/train_placenames_by_len.npy", np.array(train_names_by_length, dtype="object"))
np.save("processed/test_placenames_by_len.npy", np.array(test_names_by_length, dtype="object"))

# # save by region
# totalnames = 0
# for name in region_name_set:
#     names_arr = names_by_length[name]
#     np.random.shuffle(names_arr)
#     Path(f"processed/{name}").mkdir(exist_ok=True)
#
#     VAL_PCT = 0.1
#     val_size = int(len(names_arr) * VAL_PCT)
#
#     train_names = names_arr[:-val_size]
#     test_names = names_arr[-val_size:]
#
#     np.save(f"processed/{name}/placenames.train.npy", train_names)
#     np.save(f"processed/{name}/placenames.test.npy", test_names)
#     print(f"\n{name}: {len(names_arr)} entries; (Test: {len(test_names)}, Train: {len(train_names)})")
#     totalnames += len(names_arr)
