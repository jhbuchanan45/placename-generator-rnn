import torch
import torch.nn as nn
import string
import numpy as np

device = "cpu"


def round_eight(num):
    return num + (8 - (num % 8))


region_names = list(np.load("processed/region_names.npy", allow_pickle=True))
n_regions = round_eight(len(region_names))

all_letters = string.ascii_letters + " -'"
n_letters = round_eight(len(all_letters) + 1)


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


def index_to_letter(index):
    if index >= len(all_letters) + 1:
        return None
    else:
        return all_letters[index]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_regions):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_regions + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_regions + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, region, input, hidden):
        combined_input = torch.cat((region, input, hidden), 1)
        hidden = self.i2h(combined_input)
        output = self.i2o(combined_input)
        combined_output = torch.cat((hidden, output), 1)
        output = self.o2o(combined_output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        return torch.zeros(batch_size, self.hidden_size, device=device)
