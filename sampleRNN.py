import rnn as rnn_impl

import torch

max_length = 20

rnn = rnn_impl.RNN(rnn_impl.n_letters, 256, rnn_impl.n_letters, rnn_impl.n_regions)
rnn.load_state_dict(torch.load("model-placenames-gb-1631560872.pth"))


# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        region_tensor = rnn_impl.region_to_tensor(category)
        input = rnn_impl.placename_to_input_tensor(start_letter, len(start_letter))
        hidden = rnn.init_hidden(1, "cpu").view(-1, rnn.hidden_size)

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(region_tensor.view(-1, rnn_impl.n_regions), input.view(-1, rnn_impl.n_letters), hidden)
            # print(output)
            topv, topi = output.topk(1)
            topi = topi[0]
            if topi >= len(rnn_impl.all_letters):
                break
            else:
                letter = rnn_impl.all_letters[topi]
                output_name += letter
            input = rnn_impl.placename_to_input_tensor(letter, len(letter))

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


print(rnn_impl.region_names[9])
samples(rnn_impl.region_names[9])
