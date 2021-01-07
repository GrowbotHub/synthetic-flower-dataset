"""
Copy files from the batches on the local disk to the main folder on server.
"""
import shutil
import os
destination = "/Users/gil/Desktop/mnt/proj9/cvlabsrc1/cvlab/datasets_hugonot/project_emna_gil_flowers/natural"
source_dir = "./outputs/batch_"
percent = "%"
for batch in range(16,19):
    source = os.listdir(source_dir+str(batch)+"/natural")
    num_images = len(source)
    for idx, files in enumerate(source):
        print("\rBatch %d: %.3d%s done." %(batch, int(((idx+1)/num_images) * 100),percent), end='')
        if files.endswith(".png"):
            shutil.copy(source_dir+str(batch)+"/natural/"+files,destination)