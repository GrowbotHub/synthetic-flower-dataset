"""
Check consistency of the dataset in the main folder on the server.
"""
import pandas as pd
import shutil
import numpy as np
import csv
import os

# Where to look for batches
dir_prefix = "/Users/gil/Desktop/mnt/proj9/cvlabsrc1/cvlab/datasets_hugonot/project_emna_gil_flowers/"

result = pd.read_csv(dir_prefix+"flowers_dataset.csv")
rows = len(result.index)

# Check for duplicate filenames (should be none)
duplicates = result['filename'].duplicated().any()



def find_file_in_batch(index, fname):
    if not os.path.isfile(dir_prefix+"natural/"+fname):
        raise FileNotFoundError

print("Counting # of images")
directory = dir_prefix+"natural/"
images = [name for name in os.listdir(directory) if (name[-3:] == "png" and name[0] != '.')]
number_of_images = len(images)
print("Counting done")

percent = "%"
not_found_df = pd.DataFrame(columns=result.columns)

# Track progress and how many image files are not found
global_count = 0
progress = 0

for index, row in result.iterrows():
    print("\r%.3d%s done." %(int(((index+1)/rows) * 100),percent), end='')
    fname = row[0]
    try:
        find_file_in_batch(0,fname)
    except FileNotFoundError:
        global_count += 1
        not_found_df = not_found_df.append(row)

# Write not found files to csv file on disk
not_found_df.to_csv(path_or_buf="./data/notfound.csv",index=False)
print("\n", end="")

print("Done! Total not-found images: "+str(global_count))
print("Duplicate filenames in csv: "+str(duplicates))
print("Total number of rows in csv file: "+str(rows))
print("Total number of images counted  : "+str(number_of_images))