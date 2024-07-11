import time

import pickle
from sklearn.metrics import pairwise_distances
import pandas as pd
import sys 
import os 

# Load the DataFrame from the Pickle file
with open('filtered_thresholded_pairs.pkl', 'rb') as f:
    loaded_df = pickle.load(f)

# Modify the DataFrame to show only the file names
loaded_df['Image Path 1'] = loaded_df['Image Path 1'].apply(os.path.basename)
loaded_df['Image Path 2'] = loaded_df['Image Path 2'].apply(os.path.basename)

# Filter the DataFrame to show only rows where the file names are different
filtered_df = loaded_df.loc[loaded_df['Image Path 1'] != loaded_df['Image Path 2']]

print(filtered_df)
    