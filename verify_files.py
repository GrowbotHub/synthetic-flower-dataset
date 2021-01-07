"""
Check consistency of the dataset accross the batches on the local disk.
"""
import pandas as pd
import shutil
import numpy as np
import csv
import os

# Bounds of the batches to check
min_batch_number = 1
max_batch_number = 1

# Where to look for batches
dir_prefix = "./outputs/batch_"

# Load csv file from each batch and save as one
dataframes = []
for i in range(min_batch_number,max_batch_number+1):
    csv_file = dir_prefix+str(i)+"/flowers_dataset.csv"
    dataframes.append(pd.read_csv(csv_file))
result = pd.concat(dataframes,ignore_index=True)
rows = len(result.index)

# Check for duplicate filenames (should be none)
duplicates = result['filename'].duplicated().any()



def find_file_in_batch(index, fname):
    if not os.path.isfile(dir_prefix+str(index)+"/natural/"+fname):
        raise FileNotFoundError

print("Counting # of images")
number_of_images = 0
for batch_num in range(min_batch_number,max_batch_number+1):
    number_of_images += len([name for name in os.listdir(dir_prefix+str(batch_num)+"/natural") if (os.path.isfile('../3D/outputs/batch_'+str(batch_num)+"/natural/"+name) and name[-3:] == "png")])
print("Counting done")

percent = "%"
not_found_df = pd.DataFrame(columns=result.columns)

# Track progress and how many image files are not found
global_count = 0
progress = 0

for index, row in result.iterrows():
    print("\r%.3d%s done." %(int(((index+1)/rows) * 100),percent), end='')
    batch_index = min_batch_number
    found = False
    fname = row[0]
    while not found and batch_index <= max_batch_number:
        try:
            find_file_in_batch(batch_index,fname)
            found = True
        except FileNotFoundError:
            batch_index += 1
            if batch_index > max_batch_number:
                global_count += 1
                not_found_df = not_found_df.append(row)

# Write not found files to csv file on disk
not_found_df.to_csv(path_or_buf="notfound.csv",index=False)
print("\n", end="")

print("Done! Total not-found images: "+str(global_count))
print("Duplicate filenames in csv: "+str(duplicates))
print("Total number of rows in csv file: "+str(rows))
print("Total number of images counted  : "+str(number_of_images))