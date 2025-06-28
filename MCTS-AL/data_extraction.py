import numpy as np
import os

def data_extraction(rdnum):
    data_dir = os.getcwd() + f'/data/Round{rdnum-1}'
    if not os.path.exists(data_dir):
        raise ValueError('Missing data location')
    
    input_x = np.load(data_dir + "/volume mat.npy", allow_pickle=True)
    input_y = np.load(data_dir + "/modulus.npy", allow_pickle=True)
    
    return input_x, input_y
    
    

