
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import matplotlib.pyplot as plt


def min_max_scale_data(data):
    """
    Scales the X, Y, Z coordinates of a 3D tensor using min-max scaling, transforming the values 
    to a range between 0 and 1.

    Args:
        data (torch.Tensor): A tensor of shape (N, 3), where N is the number of points, and each 
                             point consists of X, Y, Z coordinates.

    Returns:
        torch.Tensor: A tensor of the same shape (N, 3) with the X, Y, Z coordinates scaled to 
                      the range [0, 1].
    """

    data = data.cpu().numpy()
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))

    x = scaler.fit_transform(x.reshape(-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1))
    z = scaler.fit_transform(z.reshape(-1, 1))

    return torch.tensor(np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1))


def rotate_face(data):
    """
    Rotates the 3D coordinates of a face so that the face is aligned to a certain orientation.

    Args:
        pts (list): A list containing 3D point coordinates as a tensor. The first element (pts[0]) 
                    is expected to be a 3D array where each row represents the X, Y, Z coordinates
                    of a facial landmark.

    Returns:
        torch.Tensor: A tensor with the rotated facial points
    """
    x = data[0, :]
    y = data[1, :]
    z = data[2, :]

    face_tensor = torch.stack([x, y, z], dim=1).to('cuda:0')
    # Väldigt hustler-lösning, får inte skiten att se bra ut

    sum_vectors = torch.sum(face_tensor, dim=0)
    centroid = sum_vectors / face_tensor.shape[0]
    face_tensor_orgin = (face_tensor - centroid).cpu().numpy()


    P1 = face_tensor_orgin[0]
    P2 = face_tensor_orgin[16]

    y1 = P1[1]
    y2 = P2[1]

    while abs(y1 - y2) > 1:
        r = R.from_euler('z', -0.1, degrees=True)
        face_tensor_orgin =  face_tensor_orgin @ r.as_matrix().T
        P1 = face_tensor_orgin[0]
        P2 = face_tensor_orgin[16]
        y1 = P1[1]
        y2 = P2[1]



    ### VET INTE OM DETTA FUNKAR ###
    z1 = P1[2]
    z2 = P2[2]
    x = face_tensor_orgin[33][0]

    i = 0.1
    if x > 0:
        i = -0.1

    while abs(z1 - z2) > 1:
        r = R.from_euler('y', i, degrees=True)
        face_tensor_orgin =  face_tensor_orgin @ r.as_matrix().T
        P1 = face_tensor_orgin[0]
        P2 = face_tensor_orgin[16]
        z1 = P1[2]
        z2 = P2[2]

    ### VET INTE OM DETTA FUNKAR SLUT ###

    if face_tensor_orgin[25][1] > 0:
        r = R.from_euler('z', 180, degrees=True)
        face_tensor_orgin =  face_tensor_orgin @ r.as_matrix().T
    
    
    
    return torch.tensor(face_tensor_orgin)

def fft_feature_vector(face_tensor):
    """
    Computes the Fast Fourier Transform (FFT) of a 3D input array and returns the magnitude 
    of the transformed array, excluding the first row of each column.

    The function first applies the N-dimensional FFT (fftn) on the input array, computes 
    the magnitude of the complex result, and then removes the first row of the magnitude 
    values. It only keeps the first three columns of the magnitude array for further processing.

    Args:
        arr (np.ndarray): A 3D numpy array of shape (N, M) on which the FFT is to be performed.

    Returns:
        np.ndarray: A 2D numpy array of shape (N-1, 3) containing the magnitude values of 
                    the transformed array, with the first row of each column removed.
    """

    transformed = np.fft.fftn(face_tensor)

    magnintude = np.abs(transformed)

    transformed_array = np.zeros(shape=(magnintude.shape[0]-1, magnintude.shape[1]))
    for i in range(0,3):

        column = magnintude[:,i][1:]
        transformed_array[:,i] = column

    return transformed_array

def plot_3d_2d_scatter(face_tensor_origin):
    # Extract the x, y, z coordinates
    x = face_tensor_origin[:, 0]  # First column (x)
    y = face_tensor_origin[:, 1]  # Second column (y)
    z = face_tensor_origin[:, 2]  # Third column (z)

    # 3D scatter plot
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.scatter(x, y, z)
    ax_3d.set_xlabel('X axis')
    ax_3d.set_ylabel('Y axis')
    ax_3d.set_zlabel('Z axis')
    plt.title('3D Scatter Plot')

    # 2D scatter plot
    fig_2d = plt.figure()
    plt.scatter(x, y)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('2D Scatter Plot')

    # Show both plots
    plt.show()
