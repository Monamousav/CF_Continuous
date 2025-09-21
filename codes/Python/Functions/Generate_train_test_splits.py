# %%
## === SETUP: Load required libraries ===
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import sys
#import tensorflow as tf

import pandas as pd
import sklearn as sk
import platform
import math
import pyreadr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pyreadr


from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

print(f"Python Platform: {platform.platform()}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if torch.cuda.is_available() else "NOT AVAILABLE")


# %%
# === set directory ===
#import os
#notebook_dir = "/Users/mmousavi2/Dropbox/Causal_climate/Data_and_results_CF_Continuous"  
#os.chdir(notebook_dir)  # Change directory
#print(os.getcwd())  

# === set directory ===
from pathlib import Path
import os

# project root = two levels up from this script (../CF_Continuous)
project_root = Path(__file__).resolve().parents[3]
notebook_dir = project_root  # CF_Continuous root

os.chdir(notebook_dir)
print("Working dir:", os.getcwd())


# %%
# === LOAD DATA ===
data_2nd_stage = pyreadr.read_r('./data/data_20230504/data_2nd_stage.rds')[None]
evall_N_seq = pyreadr.read_r('./data/data_20230504/evall_N_seq.rds')[None]
sim_ids_all = np.sort(data_2nd_stage['sim'].unique())

# %%
# === GENERATE TRAIN-TEST SPLITS ===
def create_split_csv(n_fields, output_path):
    np.random.seed(42)
    splits = []
    for j in range(len(sim_ids_all)):
        test_id = sim_ids_all[j]
        possible_train_ids = np.delete(sim_ids_all, j)
        train_ids = np.random.choice(possible_train_ids, n_fields, replace=False)
        split = {'test_id': test_id}
        split.update({f'train_{i+1}': train_ids[i] for i in range(n_fields)})
        splits.append(split)
    df = pd.DataFrame(splits)
    df.to_csv(output_path, index=False)

from pathlib import Path
import os

def maybe_create_split(n_fields, filename):
    if not Path(filename).exists():
        create_split_csv(n_fields, filename)
        print(f"✅ Created {filename}")
    else:
        print(f"⚡ Already exists: {filename}")



os.makedirs("./data/train_test_split", exist_ok=True)

# list of splits
field_counts = [1,2,3,4, 5, 10, 20]


# loop over them
for n in field_counts:
    filename = f'./data/train_test_split/train_test_splits_{n}fields.csv'
    maybe_create_split(n, filename)

# %%



