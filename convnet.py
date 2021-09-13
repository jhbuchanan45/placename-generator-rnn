# the wrong approach, cnn architectures are not suitable (at least in this form) for name generation problems
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import random

MODEL_NAME = f"model-placenames-gb-{int(time.time())}"
print(MODEL_NAME)

all_letters = string.ascii_letters + " -'"
n_letters = len(all_letters) + 1

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU running")
else:
    device = torch.device("cpu")
    print("CPU running")

zero_tensor = torch.zeros(1, n_letters).view(-1, 1, n_letters)


def load_data():
    training_data_dict = dict()
    testing_data_dict = dict()
    region_names = list(np.load("processed/region_names.npy", allow_pickle=True))

    for name in region_names:
        training_data_dict[name] = np.load(f"processed/{name}/placenames.train.npy", allow_pickle=True)
        testing_data_dict[name] = np.load(f"processed/{name}/placenames.test.npy", allow_pickle=True)

    return training_data_dict, testing_data_dict, region_names


training_data_dict, testing_data_dict, region_names = load_data()


def region_to_tensor(region):
    li = region_names.index(region)
    tensor = torch.zeros(1, len(region_names))
    tensor[0][li] = 1
    return tensor


def placename_to_input_tensor(placename):
    tensor = torch.zeros(len(placename), 1, n_letters)
    for li in range(len(placename)):
        letter = placename[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


def placename_to_target_tensor(placename):
    letter_indexes = [all_letters.find(placename[li]) for li in range(1, len(placename))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def stored_to_tensor(stored_placename):
    region_tensor = region_to_tensor(stored_placename[2])
    placename_tensor = placename_to_input_tensor(stored_placename[0])

    input_tensor = (region_tensor, placename_tensor)
    target_tensor = placename_to_target_tensor(stored_placename[0])
    return input_tensor, target_tensor


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

    for i in range(0, len(training_list_cur), 2):
        training_data_mixed.append(to_tensor(list(training_list_cur[i])))

    testing_data_mixed += [append_region(list(data)) for data in testing_data_dict[key]]

    training_data_dict[key] = False
    testing_data_dict[key] = False
    print(len(training_data_mixed))

del training_data_dict
del testing_data_dict
np.random.shuffle(training_data_mixed)
np.random.shuffle(testing_data_mixed)


# Random item from a list
def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_pair():
    region = random_choice(region_names)
    placename = random_choice(training_data_dict[region])
    return region, placename


# get placename from random region
def random_training_tensor_combo():
    region, placename = random_training_pair()
    region_tensor = region_to_tensor(region)
    input_tensor = placename_to_input_tensor(placename)
    target_tensor = placename_to_target_tensor(placename)
    return region_tensor, input_tensor, target_tensor


class Net(nn.Module):
    def __init__(self, letter_history=3):
        super().__init__()
        self.letter_history = letter_history
        self.conv1 = nn.Conv1d(1 + self.letter_history, 32, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.conv3 = nn.Conv1d(64, 256, 2)

        x = torch.randn(1 + self.letter_history, n_letters).view(-1, 1 + self.letter_history, n_letters)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, n_letters)

    def convs(self, input_tensor):
        input_tensor = F.max_pool1d(F.relu(self.conv1(input_tensor)), 3)
        input_tensor = F.max_pool1d(F.relu(self.conv2(input_tensor)), 3)
        input_tensor = F.relu(self.conv3(input_tensor))

        if self._to_linear is None:
            self._to_linear = input_tensor[0].shape[0] * input_tensor[0].shape[1]
        return input_tensor

    def forward(self, region_tensor, placename_tensor):
        input_tensor = torch.cat((placename_tensor, region_tensor), 1)

        input_tensor = self.convs(input_tensor)
        input_tensor = input_tensor.view(-1, self._to_linear)
        input_tensor = F.relu(self.fc1(input_tensor))
        input_tensor = self.fc2(input_tensor)
        return F.softmax(input_tensor, dim=1)


def fwd_pass(input_tuple_batch, train=False):
    if train:
        net.zero_grad()

    loss = 0

    for placename_tuple in input_tuple_batch:
        t_region, t_input = placename_tuple[0]
        t_target = placename_tuple[1].to(device)
        t_input = torch.cat((zero_tensor, zero_tensor, t_input), 0).to(device)

        for j in range(t_target.size(0)):
            word_pos = j + net.letter_history - 1
            input_slice = t_input[j:word_pos + 1]

            output = net(
                F.pad(input=t_region, pad=(0, n_letters - len(region_names))).view(-1, 1, n_letters).to(device),
                input_slice.view(-1, net.letter_history, n_letters))
            loss += loss_function(output, t_target[j].view(-1))

    if train:
        loss.backward()
        optimizer.step()
    return loss


net = Net().to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001)
loss_function = nn.NLLLoss()


def train():
    BATCH_SIZE = 64
    EPOCHS = 2

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(training_data_mixed), BATCH_SIZE)):
                batch = training_data_mixed[i:i + BATCH_SIZE]

                loss = fwd_pass(batch, train=True)
                if i % 50 == 0:
                    val_loss = test(size=100)
                    f.write(
                        f"{MODEL_NAME}, {round(time.time(), 3)}, {round(float(loss), 4)}, {round(float(val_loss), 4)}\n"
                    )


def test(size=32):
    random_start = np.random.randint(len(testing_data_mixed) - size)

    val_loss = fwd_pass([stored_to_tensor(list(data)) for data in testing_data_mixed[random_start:random_start + size]])

    return val_loss


train()
