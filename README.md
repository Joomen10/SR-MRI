# SR-MRI

# SR-MRI

This repository contains code and resources for a **super-resolution MRI** project using a 2D U-Net baseline. The primary goal is to enhance the resolution of MRI scans by reconstructing high-quality 2D slices, which can then be reassembled into isotropic 3D volumes.

---


- **`data/`**: Holds the input MRI data, intermediate results (upsampled volumes), and extracted 2D slices.  
- **`models/`**: Contains model definitions. Here, `model.py` defines the 2D U-Net (`UNet2D`) using PyTorch.  
- **`train.py`**: Main training script that loads 2D slices, builds a `DataLoader`, and trains the U-Net model. Saves the trained weights to `.pth`.  
- **`inference.py`**: Applies the trained U-Net to a new low-resolution (upsampled) 3D volume, processing it slice-by-slice and saving the super-resolved output.  
- **`train.ipynb` / `inference.ipynb`**: Jupyter notebooks for interactive experiments or demos of training and inference.  
- **`superres_unet_v4.pth`**: A saved model checkpoint containing trained weights for the U-Net.  
- **`train.sh`**: A batch submission script for training on an HPC server (e.g., NYU HPC).  

---

## Usage

1. **Preprocessing (Optional)**  
   - If needed, use a script or notebook to resample and extract 2D slices from your LR MRI volumes, then place the resulting `.npy` files in `data/`.

