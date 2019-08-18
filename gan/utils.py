import pickle 
import numpy as np 
import matplotlib.pyplot as plt 


def load_data(file):
    with open(file, 'rb') as outfile:
        results = pickle.load(outfile)
    return outfile 




def read_loaded(outfile):
    for el in range(outfile):
        for _iter in range(len(el)):
            plt.imshow(outfile[el][_iter].detach().cpu().numpy())

def main():
    outfile = load_data('test')
    read_loaded(outfile)



if __name__ == "__main__":

    main()
