#############################################################################################

# Importing packages
import os
import argparse
import json
import numpy as np
from random_motion_complex_change_fourier import *
import h5py
import matplotlib.pyplot as plt
import torchio as tio

# Fourier Transforms Functions
def ifft2(a, axes=(-2, -1), norm=None):
    a = np.fft.ifftshift(a, axes=axes)
    a = np.fft.ifft2(a, axes=axes, norm=norm)
    a = np.fft.fftshift(a, axes=axes)
    return a

def fft2(a, axes=(-2, -1), norm=None):
    a = np.fft.ifftshift(a, axes=axes)
    a = np.fft.fft2(a, axes=axes, norm=norm)
    a = np.fft.fftshift(a, axes=axes)
    return a

def get_split_data(input_path):
    """Split the data into torchio.Subjects (real and imaginary parts split).
    Parameters
    ----------
    input_path : str
        Path to the h5 file.

    Returns
    -------
    real_subject : torchio.Subject
        Real part of the data.
    imag_subject : torchio.Subject
        Imaginary part of the data.
    """

    # Read the data
    d = h5py.File(input_path, 'r')
    original_kspace = d["kspace"][:,:,:,:].copy()
    d.close()

    # Get the real and imaginary parts
    ispace_original = ifft2(original_kspace, norm='ortho').astype(np.complex128)
    ispace_dimension_good = np.transpose(ispace_original, (1, 3, 2, 0))

    # Create TorchIO subjects for the real and imaginary data
    real_subject = tio.Subject(ispace=tio.ScalarImage(tensor=np.real(ispace_dimension_good)))
    imag_subject = tio.Subject(ispace=tio.ScalarImage(tensor=np.imag(ispace_dimension_good)))

    return real_subject, imag_subject

def corrupt_ispace(real_subject, imag_subject, rotation_degrees=(-10,10), translation=(-10,10), 
                   image_interpolation='bspline'):
    """
    Corrupt the data by applying a random motion to the real and imaginary parts of the data.

    Parameters
    ----------
    real_subject : torchio.Subject
        Real part of the data.
    imag_subject : torchio.Subject
        Imaginary part of the data.
    rotation_degrees : tuple
        Tuple of the minimum and maximum rotation degrees.
    translation : tuple
        Tuple of the minimum and maximum translation.
    image_interpolation : str
        Interpolation method to use.

    Returns
    -------
    corrupted_kspace : np.array
        Corrupted kspace.
    """

    # Apply the RandomMotion transform to have the random parameters of the motion
    random_motion_params = RandomMotion(degrees=rotation_degrees, translation=translation, \
                                        image_interpolation='bspline').apply_transform(\
                                            real_subject)

    motion = Motion(**random_motion_params)

    real_motion_subject = motion(real_subject)
    imag_motion_subject = motion(imag_subject)

    corrupted_ispace = (real_motion_subject['ispace'].data.numpy() + 1j* 
                        imag_motion_subject['ispace'].data.numpy())

    corrupted_kspace = fft2(np.transpose(corrupted_ispace, (3, 0, 2, 1)), norm='ortho').astype(np.complex128)

    return corrupted_kspace

def save_h5_corrupted_kspace(corrupted_kspace_path, corrupted_kspace):
    """ Save the corrupted kspace in a h5 file.

    Parameters
    ----------
    corrupted_kspace_path : str
        Path to the h5 file.
    corrupted_kspace : np.array
        Corrupted kspace.

    Returns
    -------
    None
    """

    with h5py.File(corrupted_kspace_path, "w") as f:
        temp_corr_kspace = f.create_dataset(name="kspace", data=corrupted_kspace)
    f.close()

def main(data_dir):

    # base directory
    BASE = data_dir
    data_to_original_dict = {}

    # loop through all the subdirectories
    for main_dirs in os.listdir(BASE):
        for multicoil_dirs in os.listdir(BASE + "/" + main_dirs):
            # ignore files
            if ('.' in multicoil_dirs):
                continue
            # for each file in the subdirectory
            for file in os.listdir(BASE + main_dirs + "/" + multicoil_dirs):
                # right file types
                if ('.h5' in file):
                    # do the corruption and save
                    real_data, imag_data = get_split_data(BASE + "/" +main_dirs + "/" + multicoil_dirs + "/" + file)
                    corrupted_kspace = corrupt_ispace(real_data, imag_data)
                    corrupted_kspace_path = BASE + main_dirs + "/" + multicoil_dirs + "/" + file[:-3] + "_corrupted.h5"
                    save_h5_corrupted_kspace(corrupted_kspace_path, corrupted_kspace)

                    # dictionary to map to original data
                    data_to_original_dict['/data/' + multicoil_dirs + '/' + file] = '/data/' + multicoil_dirs + '/' + file
                    data_to_original_dict['/data/' + multicoil_dirs + '/' + file[:-3] + '_corrupted.h5'] = '/data/' + multicoil_dirs + '/' + file

    # save the dictionary
    with open(BASE + 'data_to_original_dict.json', 'w') as json_file:
        json.dump(data_to_original_dict, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create corrupted data with motion artifacts')
    parser.add_argument(
        '--data_dir', required=True,
        help='directory that conatins the uncorrupted data')
    args = parser.parse_args()
    main(args.data_dir)