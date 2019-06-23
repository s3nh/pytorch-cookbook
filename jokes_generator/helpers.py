import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable


# Reading all data



all_characters = string.printable
n_characters = len(all_characters)


def read_file(filename):

    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)




def time_since(since):

    s = time.time() - since
    m = math.floor(s/60)
    s -= m * 60
    return '{} {}'.format(m, s)
