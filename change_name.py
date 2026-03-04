import os 
import shutil

data_path = "/data/zhouzhexin/ctdsa/extracted_data_shift_only"

file_list = os.listdir(data_path)

for subject_id in file_list:
    if subject_id == "subject06":
        output_path = "/data/zhouzhexin/ctdsa/data/test_shift_only"
    else:
        output_path = "/data/zhouzhexin/ctdsa/data/train_shift_only"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "drr"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "img"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "seg"), exist_ok=True)
    drr_path = os.path.join(data_path, subject_id, "diffdrr")
    for file in os.listdir(drr_path):
        if file.startswith("drr"):
            new_name = subject_id + "_" + file
            shutil.move(os.path.join(drr_path, file), os.path.join(output_path,"drr", new_name))
        elif file.startswith("mask"):
            new_name = subject_id + "_" + file
            shutil.move(os.path.join(drr_path, file), os.path.join(output_path,"seg", new_name))