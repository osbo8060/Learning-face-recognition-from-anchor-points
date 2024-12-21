# ==============================================================
#                   Made by Oscar Boman
# ==============================================================

# %%
from facetorch import FaceAnalyzer
from facetorch.analyzer.utilizer import LandmarkDrawerTorch
from omegaconf import OmegaConf
from torch.nn.functional import cosine_similarity
from typing import Dict
import operator
import torchvision
import os
import shutil
import torch
from PIL import Image
import argparse

# %% 
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=list, default=['../Dataset/Labeled_Faces_in_the_Wild/'])

args = parser.parse_args()
# %%

DATASET_PATHS = ['../Dataset/Labeled_Faces_in_the_Wild/']
NEW_PATH = './Curated_Dataset/'


if os.path.exists(NEW_PATH):
    shutil.rmtree(NEW_PATH)
os.makedirs(NEW_PATH)

# %%
lowerthresholdImages = 20
higherthresholdImages = 30

for dataset_path in DATASET_PATHS:
    for name in os.listdir(dataset_path):

        temp_path = os.path.join(dataset_path, name)
        print(dataset_path)
        if len(os.listdir(temp_path)) >= lowerthresholdImages:
            temp_new_path = os.path.join(NEW_PATH, name)
            images = os.listdir(temp_path)
            if os.path.exists(temp_new_path):
                shutil.rmtree(temp_new_path)
            os.makedirs(temp_new_path, exist_ok=True)

            # Only copy the first x images, to combat class inbalance
            for image in images[:higherthresholdImages]:
                src = os.path.join(temp_path, image)
                dst = os.path.join(temp_new_path, image)
                shutil.copy2(src, dst)


# ==============================================================
#                   Made By Daniel Cada
# ==============================================================
# %%
"TODO OBS!! Nuvarande problem med att om man kör flera gånger så skapas '_hor_flipped_hor_flipped' bilder av ngn anledning. Inte klart varför"

"""
    Flips an Image on it's horizontal (x) axis and saves it with it's name + "_hor_flipped"
    Ex: "path/image.jpg" -> "path/image_x_flipped.jpg"

    Args:
        path_to_image (str): The file path to the image to flip.

    """
def flipImageHorizontalAndSave(path_img):
    image_name = path_img[:-4]
    newName = image_name + "_hor_flipped.jpg"
    if not os.path.isfile(newName):
        img = Image.open(path_img)
        flip_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flip_img.save(newName)

## FLIPS ALL IMAGES AFTER FILTER, BEFORE ANCHOR POINTS
selectedPath = NEW_PATH

## FLIPS ALL IMAGES AFTER ANCHOR POINTS
# selectedPath = "Pre-processing\dataset\face_dataset"

for name in os.listdir(selectedPath):
    temp_path = selectedPath + f'{name}'
    for image in os.listdir(temp_path):
        path_name = temp_path +"/"+ image
        flipImageHorizontalAndSave(path_name)

