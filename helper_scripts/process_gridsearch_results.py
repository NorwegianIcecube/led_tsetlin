import pandas as pd

# read the data from the training.csv file that is one level above this file
import os

path = "C:/Users/sigur/Git/led_tsetlin/training.csv"
current_dir = os.getcwd()
print(current_dir)
if os.access(path, os.R_OK):
    print("File has read permission")
else:
    print("File does not have read permission")

data = pd.read_csv(filepath_or_buffer=path)
print(data)
