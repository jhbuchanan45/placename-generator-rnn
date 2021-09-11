# see https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#exercises

import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
from rnn import RNN, region_names, n_regions

MODEL_NAME = f"model-placenames-gb-{int(time.time())}"
print(MODEL_NAME)


def round_eight(num):
    return num + (8 - (num % 8))


all_letters = string.ascii_letters + " -'"
n_letters = round_eight(len(all_letters) + 1)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU running")
else:
    device = torch.device("cpu")
    print("CPU running")


def load_data():
    training_data_dict = dict()
    testing_data_dict = dict()

    for name in region_names:
        training_data_dict[name] = np.load(f"processed/{name}/placenames.train.npy", allow_pickle=True)
        testing_data_dict[name] = np.load(f"processed/{name}/placenames.test.npy", allow_pickle=True)

    return training_data_dict, testing_data_dict, region_names


training_data_dict, testing_data_dict, region_names = load_data()


def region_to_tensor(region):
    li = region_names.index(region)
    tensor = torch.zeros(n_regions, device=device)
    tensor[li] = 1
    return tensor


def placename_to_input_tensor(placename, max_len):
    tensor = torch.zeros(max_len, n_letters, device=device)
    for li in range(len(placename)):
        letter = placename[li]
        tensor[li][all_letters.find(letter)] = 1
    return tensor


def placename_to_target_tensor(placename, max_len):
    letter_indexes = [all_letters.find(placename[li]) for li in range(1, len(placename))]
    letter_indexes.append(len(all_letters))  # EOS
    letter_indexes = np.pad(letter_indexes, (0, max_len - len(letter_indexes)), mode='constant')
    return torch.tensor(letter_indexes, device=device)


def stored_to_tensor(stored_placename, max_len):
    region_tensor = region_to_tensor(stored_placename[2])
    input_tensor = placename_to_input_tensor(stored_placename[0], max_len=max_len)
    target_tensor = placename_to_target_tensor(stored_placename[0], max_len=max_len)
    return region_tensor, input_tensor, target_tensor


def batch_to_tensor(batch):
    max_name = max(np.concatenate(batch[:, 0:1]), key=len)
    max_len = len(max_name)

    t_regions, t_inputs, t_targets = list(), list(), list()

    for placename in batch:
        t_region, t_input, t_target = stored_to_tensor(placename, max_len=max_len)

        t_regions.append(t_region)
        t_inputs.append(t_input)
        t_targets.append(t_target)

    return torch.stack(t_regions, dim=0), torch.stack(t_inputs, dim=0), torch.stack(t_targets, dim=0)


training_data_mixed, testing_data_mixed = list(), list()

for key in training_data_dict:

    def append_region(data):
        data.append(key)
        return data


    def to_tensor(data):
        append_region(data)
        return stored_to_tensor(data)


    training_list_cur = training_data_dict[key]
    print(len(training_list_cur))

    for i in range(0, len(training_list_cur), 1):
        training_data_mixed.append(append_region(list(training_list_cur[i])))

    testing_data_mixed += [append_region(list(data)) for data in testing_data_dict[key]]

    training_data_dict[key] = False
    testing_data_dict[key] = False

del training_data_dict
del testing_data_dict
training_data_mixed = np.array(training_data_mixed)
testing_data_mixed = np.array(testing_data_mixed)
np.random.shuffle(training_data_mixed)
np.random.shuffle(testing_data_mixed)

rnn = RNN(n_letters, 256, n_letters, n_regions).to(device)

loss_func = nn.NLLLoss()
learning_rate = 0.0005


def fwd_pass(region_tensor, input_tensor, target_tensor, train=False):
    # target_tensor = torch.flip(torch.rot90(target_tensor, -1), dims=[1])

    # input_tensor = torch.flip(torch.rot90(input_tensor, -1), dims=[1])
    hiddens = rnn.init_hidden(region_tensor.size(0), device)

    if train:
        rnn.zero_grad()

    loss = 0

    with torch.cuda.amp.autocast():
        for i in range(input_tensor.size(1)):
            outputs, hiddens = rnn(region_tensor, input_tensor[:, i:i + 1].squeeze_(1), hiddens)
            target = target_tensor[:, i:i + 1]

            target = target.squeeze(-1)

            l = loss_func(outputs, target)
            loss += l

    if train:
        loss.backward()

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

    return loss.item() / input_tensor.size(1)


def batch_pass(batch, train=False):
    batch_loss = 0

    t_regions = batch[0]
    t_inputs = batch[1]
    t_targets = batch[2]

    batch_loss += fwd_pass(t_regions, t_inputs, t_targets, train=train)

    return batch_loss / len(batch)


def batch_tuples_to_tensors(batch):
    batch_regions = torch.from_numpy(batch[:, 0:1])
    batch_inputs = torch.from_numpy(batch[:, 1:2])
    batch_targets = torch.from_numpy(batch[:, 2:3])
    return batch_regions, batch_inputs, batch_targets


def get_batches(dataset, batch_size):
    for i in range(0, dataset.shape[0] - batch_size, batch_size):
        cur_batch = dataset[i:i + batch_size, :]

        cur_batch = batch_to_tensor(cur_batch)

        yield cur_batch


def train():
    BATCH_SIZE = round_eight(192)
    EPOCHS = 10

    with open("model.log", "a") as f:

        for epoch in range(EPOCHS):
            batches = get_batches(training_data_mixed, BATCH_SIZE)

            for i in tqdm(range(0, training_data_mixed.shape[0] - BATCH_SIZE, BATCH_SIZE)):
                batch = next(batches)

                train_loss = batch_pass(batch, train=True)
                if i % 15 == 0:
                    test_loss = test(size=100)
                f.write(
                    f"{MODEL_NAME}, {round(time.time(), 3)}, {round(float(train_loss), 4)}, {round(float(test_loss), 4)}\n"
                )


def test(size=32):
    random_start = np.random.randint(len(testing_data_mixed) - size)

    test_batch = get_batches(testing_data_mixed[random_start:random_start + size], size - 1)

    test_loss = batch_pass(next(test_batch))

    return test_loss


train()
torch.save(rnn.state_dict(), f"{MODEL_NAME}.pth")
