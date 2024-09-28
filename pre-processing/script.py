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

# %%
DATASET_PATH = '../original_dataset/Labeled_Faces_in_the_Wild/lfw-deepfunneled/lfw-deepfunneled/'
ANCHOR_PATH = './dataset/anchor_points_dataset/'
FACE_PATH = './dataset/face_dataset/'

if os.path.exists(FACE_PATH):
    shutil.rmtree(FACE_PATH)

if os.path.exists(ANCHOR_PATH):
    shutil.rmtree(ANCHOR_PATH)

os.makedirs(FACE_PATH)
os.makedirs(ANCHOR_PATH)


cfg = OmegaConf.load("gpu.config.yml")
# %%
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
        fix_img_size=cfg.fix_img_size,
        return_img_data=cfg.return_img_data,
        include_tensors=cfg.include_tensors,
    )
    return response


# %%
for name in os.listdir(DATASET_PATH):

    temp_path = DATASET_PATH + f'{name}'  
    temp_anchor_path = ANCHOR_PATH + f'{name}'
    temp_face_path = FACE_PATH + f'{name}'

    if os.path.exists(temp_anchor_path):
        shutil.rmtree(temp_anchor_path)

    if os.path.exists(temp_face_path):
        shutil.rmtree(temp_face_path)
    
    os.makedirs(temp_anchor_path)
    os.makedirs(temp_face_path)

    for image in os.listdir(temp_path):

        response = Process_Image( temp_path + f'/{image}')
        pil_image = torchvision.transforms.functional.to_pil_image(response.img)
        pts = [face.preds["align"].other["lmk3d"].cpu() for face in response.faces]
        if len(pts) == 1:
            torch.save(pts[0], temp_anchor_path + f'/{image}.pt')
            pil_image.save(f'{temp_face_path}/{name}.jpg')





    


