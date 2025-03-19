# lr_preprocess.py

import os
import numpy as np
import nibabel as nib
import torchio as tio

def resample_nii(lr_path, hr_path, output_path):
    """
    Resample the LR image to match the voxel spacing of the HR image
    using TorchIO, then save the resampled image.
    """
    lr_image = tio.ScalarImage(lr_path)
    hr_image = tio.ScalarImage(hr_path)

    # Create the resample transform to match HR voxel spacing
    resample_transform = tio.Resample(target=hr_image)
    lr_upsampled = resample_transform(lr_image)

    # Save the upsampled image
    lr_upsampled.save(output_path)
    print(f"Saved upsampled LR image to: {output_path}")

def load_nii_as_numpy(filepath):
    """Load a NIfTI file and return as a NumPy array."""
    nii = nib.load(filepath)
    data = nii.get_fdata()
    return data

def extract_2d_slices(lr_path, hr_path):
    """
    Load the LR and HR volumes, assume they are already aligned
    and have the same shape in 3D (after resampling).
    Return two lists of 2D NumPy arrays: slices_lr and slices_hr.
    """
    lr_data = load_nii_as_numpy(lr_path)
    hr_data = load_nii_as_numpy(hr_path)

    # Check shapes
    if lr_data.shape != hr_data.shape:
        raise ValueError("LR and HR volumes do not have the same shape. "
                         "Consider resampling or check your orientations.")

    slices_lr = []
    slices_hr = []

    depth = lr_data.shape[0]  # e.g. [D, H, W]
    for i in range(depth):
        lr_slice = lr_data[i, :, :]
        hr_slice = hr_data[i, :, :]
        slices_lr.append(lr_slice)
        slices_hr.append(hr_slice)

    return slices_lr, slices_hr

def main():
    # Example usage
    # Paths: adjust to your file locations
    # LR files
    lr_axial_path = "../data/sub-OAS30001_ses-d0129_run-01_T1w_axial_LR.nii"
    lr_coronal_path = "../data/sub-OAS30001_ses-d0129_run-01_T1w_coronal_LR.nii"
    # HR file
    hr_path = "../data/sub-OAS30001_ses-d0129_run-01_T1w.nii"

    # Output upsampled LR file paths
    lr_axial_upsampled_path = "../data/sub-OAS30001_axial_upsampled.nii"
    lr_coronal_upsampled_path = "../data/sub-OAS30001_coronal_upsampled.nii"

    # 1) Resample LR axial image to match HR spacing
    resample_nii(lr_axial_path, hr_path, lr_axial_upsampled_path)

    # 2) Resample LR coronal image to match HR spacing
    resample_nii(lr_coronal_path, hr_path, lr_coronal_upsampled_path)

    # 3) Slice into 2D arrays
    slices_lr_axial, slices_hr_axial = extract_2d_slices(lr_axial_upsampled_path, hr_path)
    slices_lr_coronal, slices_hr_coronal = extract_2d_slices(lr_coronal_upsampled_path, hr_path)

    # Example: you can save these lists as .npy files or pass them directly.
    np.save("../data/lr_slices_axial.npy", slices_lr_axial)
    np.save("../data/hr_slices_axial.npy", slices_hr_axial)
    np.save("../data/lr_slices_coronal.npy", slices_lr_coronal)
    np.save("../data/hr_slices_coronal.npy", slices_hr_coronal)
    print("Saved 2D slices as NumPy arrays.")

if __name__ == "__main__":
    main()
