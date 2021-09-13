# see https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#exercises

import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
from rnn import RNN, region_names, n_regions, all_letters, n_letters, index_to_letter

MODEL_NAME = f"model-placenames-gb-{int(time.time())}"
print(MODEL_NAME)


def round_eight(num):
    return num + (8 - (num % 8))


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU running")
else:
    device = torch.device("cpu")
    print("CPU running")


def load_data():
    training_data_list = np.load(f"processed/train_placenames_by_len.npy", allow_pickle=True)
    testing_data_list = np.load(f"processed/test_placenames_by_len.npy", allow_pickle=True)

    return training_data_list, testing_data_list, region_names


training_data_list, testing_data_list, region_names = load_data()


def flatten_2d_list(lst):
    new_list = list()
    for sublist in lst:
        new_list.extend(sublist)
    return new_list


def get_list_for_batching(dataset):
    for words in dataset:
        np.random.shuffle(words)

    dataset = flatten_2d_list(dataset)
    print(len(dataset))

    return dataset


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
    region_tensor = region_to_tensor(stored_placename[1])
    input_tensor = placename_to_input_tensor(stored_placename[0], max_len=max_len)
    target_tensor = placename_to_target_tensor(stored_placename[0], max_len=max_len)
    return region_tensor, input_tensor, target_tensor


def batch_to_tensor(batch):
    batch = np.array(batch)
    max_name = max(np.concatenate(batch[:, 0:1]), key=len)
    max_len = len(max_name)

    t_regions, t_inputs, t_targets = list(), list(), list()

    for placename in batch:
        t_region, t_input, t_target = stored_to_tensor(placename, max_len=max_len)

        t_regions.append(t_region)
        t_inputs.append(t_input)
        t_targets.append(t_target)

    return torch.stack(t_regions, dim=0), torch.stack(t_inputs, dim=0), torch.stack(t_targets, dim=0)


learning_rate = 0.001

rnn = RNN(n_letters, 256, n_letters, n_regions).to(device)
optimiser = optim.Adam(rnn.parameters(), lr=learning_rate)
loss_func = nn.NLLLoss().to(device)


def fwd_pass(region_tensor, input_tensor, target_tensor, train=False):
    hiddens = rnn.init_hidden(region_tensor.size(0), device)

    if train:
        rnn.zero_grad()

    loss = 0

    with torch.cuda.amp.autocast():
        for i in range(input_tensor.size(1)):
            outputs, hiddens = rnn(region_tensor, input_tensor[:, i:i + 1].squeeze_(1), hiddens)
            # print(outputs.size())
            target = target_tensor[:, i:i + 1]

            target = target.squeeze(-1)

            l = loss_func(outputs, target)
            loss += l

    if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 0.5)
        optimiser.step()
        optimiser.zero_grad()

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
    for i in range(0, len(dataset) - batch_size, batch_size):
        cur_batch = dataset[i:i + batch_size]
        # print(cur_batch[0][0])

        cur_batch = batch_to_tensor(cur_batch)

        yield cur_batch


def train():
    BATCH_SIZE = round_eight(127)
    EPOCHS = 5

    with open("model.log", "a") as f:

        for epoch in range(EPOCHS):
            dataset = get_list_for_batching(training_data_list)
            batches = get_batches(dataset, BATCH_SIZE)

            for i in tqdm(range(0, len(dataset) - BATCH_SIZE, BATCH_SIZE)):
                batch = next(batches)

                train_loss = batch_pass(batch, train=True)

                if i % 15 == 0:
                    with torch.no_grad():
                        rnn.eval()
                        test_loss = test(size=128)
                        f.write(
                            f"{MODEL_NAME}, {round(time.time(), 3)}, {round(float(train_loss), 4)}, {round(float(test_loss), 4)}\n"
                        )
                        rnn.train()


testing_dataset = get_list_for_batching(testing_data_list)


def test(size=128):
    random_start = np.random.randint(len(testing_dataset) - size)

    test_batch = get_batches(testing_dataset[random_start:random_start + size], size - 1)

    test_loss = batch_pass(next(test_batch))

    return test_loss


train()
torch.save(rnn.state_dict(), f"{MODEL_NAME}.pth")
