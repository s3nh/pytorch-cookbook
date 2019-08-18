import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
import argparse


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type = str, help = 'FIle to preprocess')
    args = parser.parse_args()
    return args


def load_data(file):
    with open(file, 'rb') as outfile:
        results = pickle.load(outfile)
    return results 

def read_loaded(results):
    for el in range(len(results)):
        for iter in range(len(results[el])):
            print(iter)
            img = np.transpose(results[el][iter].detach().cpu().numpy(), (1, 2, 0))
            img = ((img+1)*255 / (2)).astype(np.uint8)
            plt.imshow(img)
            #plt.imshow(np.transpose(results[el][iter].detach().cpu().numpy(), (1, 2, 0)))
            plt.show()

def main():
    args = build_args()
    outfile = load_data(args.file)
    print(outfile)
    read_loaded(outfile)

if __name__ == "__main__":
    main()
