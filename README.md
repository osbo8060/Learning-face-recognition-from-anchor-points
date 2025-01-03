# Learning Face Recognition from Anchor Points

This project explores methods for facial recognition using anchor points extracted from facial images. By focusing on anchor points rather than raw pixel data, the project aims to emphasize the geometric and structural features of faces, providing a novel approach to facial recognition.

## Repository Structure

### Directories

#### 1. **Pre-Processing**
This directory contains scripts and notebooks for preparing the dataset and processing images to extract anchor points.

- **`curation.py`**: Curates the dataset by:
  - Filtering the number of images used per label.
  - Mirroring images to augment the dataset.

- **`process-images.py`**: Uses the `facetorch` library to extract anchor points from the curated dataset.

- **`feature_vectors.ipynb`**: Performs transformations on the extracted anchor points to create feature vectors suitable for model training.

#### 2. **Model**
This directory contains the implementation of the facial recognition models and their training/testing routines.

- **`bioNet.py`**: Defines the model architectures and the custom loss function used in training.

- **`CNN.ipynb`**: Contains the scripts for training and testing the models, including performance evaluation.

## Installation

1. Clone the repository:

2. Install dependencies:

## Usage

### Pre-Processing
1. Run `curation.py` to curate the dataset:

2. Extract anchor points using `process-images.py`:

3. Transform anchor points into feature vectors using `feature_vectors.ipynb`.

### Model Training and Testing
1. Define the model architecture in `bioNet.py`.
2. Train and test the models using `CNN.ipynb`. This notebook includes all steps for training, testing, and evaluating the model’s performance.

## Key Technologies
- **facetorch**: For extracting facial anchor points.
- **PyTorch**: For building and training machine learning models.
- **Matplotlib & NumPy**: For data visualization and numerical operations.
- **Scikit-learn**: For preprocessing and evaluation metrics.

