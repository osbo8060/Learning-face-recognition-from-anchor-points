# ==============================================================
#                   Entire File Made by Daniel Cada
# ==============================================================

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
import pandas as pd
from PIL import Image
import argparse

# 1. Create a new CSV for Anchor Points
# 2. Go inside the folder with the folders with people images
# 3. For every folder/person, check if they have 2 or more images of them
# 4. Use Facetorch to extract the Anchor Points of the people in only the first two images
# 5. Save the 68 Anchor Point Coordinates as x1,y1,z1,x2,y2,z2,...,x68,y68,z68, Label(folder_name)
# 6. Save the CSV in reasonable folder

NUMBER_OF_IMAGES = 2

ANCHOR_PATH = './Pre-processing/dataset/anchor_points_dataset/'
CSV_PATH = './Curated_Dataset/Anchor_Points.csv'

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=list, default=['./Dataset/Labeled_Faces_in_the_Wild/'])

args = parser.parse_args()
DATASET_PATHS = args.path

if not os.path.exists(ANCHOR_PATH):
    os.makedirs(ANCHOR_PATH)

cfg = OmegaConf.load("./Pre-processing/gpu.config.yml")
cfg.analyzer.use_cuda = False  # Ensure GPU is not used
cfg.analyzer.device = 'cpu'    # Explicitly set to CPU

def Process_Image(path_img):
    """
    Processes the input image and extracts important features (anchor points).

    Args:
        path_to_image (str): The file path to the image to be processed.

    Returns:
        processed_image: The image after processing 
        anchor_points: A set of key points or features detected in the image (e.g., corners, edges, or other important features).
    """
    analyzer = FaceAnalyzer(cfg.analyzer)
    response = analyzer.run(
        path_image=path_img,
        batch_size=cfg.batch_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
        fix_img_size=cfg.fix_img_size
    )
    return response

columns = [f'feature_{i}' for i in range(1, 205)] + ['label']
df_points = pd.DataFrame(columns=columns)
for DATASET_PATH in DATASET_PATHS:
    for person_folder in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_folder)
        if os.path.isdir(person_path) and len(os.listdir(person_path)) >= NUMBER_OF_IMAGES:
            images = sorted(os.listdir(person_path))
            anchor_points_array = []
            for img_file in images:
                img_path = os.path.join(person_path, img_file)
                anchor_points = Process_Image(img_path)
                pts = [face.preds["align"].other["lmk3d"].cpu() for face in anchor_points.faces]
                if len(pts) > 1:
                    print("MORE THAN ONE FACE LOCATED IN FILE: ", img_path)
                elif len(pts) == 1:
                    anchor_points_array.append(pts[0].reshape(-1).tolist())
                    # 
                    print("Face ", person_folder, " Done!")
                if len(anchor_points_array) == 2:
                    break

            if len(anchor_points_array) == 2:
                for face in anchor_points_array:
                    df_points.loc[len(df_points)] = face + [person_folder]
                print(f"Saved anchor points for {person_folder}")
            else:
                print(f"Not enough valid images for {person_folder}. Skipping.")

output_anchor_points = os.path.join(ANCHOR_PATH, "face_verification_data_points.csv")
df_points.to_csv(output_anchor_points, index=False)