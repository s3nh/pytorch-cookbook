import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T


import os 
import argparse 
import sys
import random


from model import * 
from helpers import * 


argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type = str)
argparser.add_argument('--n_epochs', type = int, default = 2000)
argparser.add_argument('--print_every', type = int, default = 100)
argparser.add_argument('--hidden_size', type = int, default = 50)
argparser.add_argument('--n_layers', type = int, default = 2)
argparser.add_argument('--learning_rate', type = float, default = 0.01)
argparser.add_argument('--chunk_len', type = int, default = 200)
args = argparser.parse_args()


file, file_len = read_file(args.filename)
print(file_len)
print(file[:50])

print(args.chunk_len)

def random_set(chunk_len):
    start_index = random.randint(0, file_len -  chunk_len)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index:end_index]
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])

    return inp, target

decoder = RNN(n_characters, args.hidden_size, n_characters, args.n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr = args.learning_rate)
criterion = nn.CrossEntropyLoss()


start = time.time()
all_losses = []
loss_avg = 0

def train(inp, target):
    target=target.unsqueeze(0)
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    print(inp)
    print(target)
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c])
    
    loss.backward()
    decoder_optimizer.step()


    return loss.data[0]/args.chunk_len


def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + 'pt'
    torch.save(decoder, save_filename)
    print("Saved as {}".format(save_filename))


try:
    print("Training for {} epochs...".format(args.n_epochs))
    for epoch in range(1, args.n_epochs +1):
        loss = train(*random_set(args.chunk_len))
        loss_avg += loss


        if epoch % args.print_every == 0:
            print('[{} ({} {}) {}]'.format(time_since(start), epoch, epoch/args.n_epochs * 100, loss))
            print(generate(decoder, 'Wh', 100), '\n')

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()



def main():

    data = load_file()

if __name__ == "__main__":

    main()
