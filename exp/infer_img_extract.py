import os
import shutil


data_dir = "/root/autodl-tmp/ppocr_data"
label_file = "/root/autodl-tmp/ppocr_data/test_paragraphs.txt"

save_dir = "./data/para_infer"
os.makedirs(save_dir, exist_ok=True)

with open(label_file, "r", encoding="utf-8") as f:
    for line in f.readlines():
        name = line.strip().split("\t")[0]
        file = os.path.join(data_dir, name)
        if os.path.exists(file):
            shutil.copyfile(
                file, 
                os.path.join(save_dir, os.path.basename(file))
            )


