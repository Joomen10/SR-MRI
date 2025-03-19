# train.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# from models.model import UNetDeep
from models.model import UNet2DDeep

class MRISuperResDataset(Dataset):
    """
    A dataset that takes lists (or arrays) of LR and HR 2D slices
    and returns them as PyTorch tensors.
    """
    def __init__(self, lr_slices, hr_slices, transform=None):
        """
        lr_slices, hr_slices: lists (or NumPy arrays) of shape (D, H, W)
        transform: any optional transform function (if needed)
        """
        self.lr_slices = lr_slices
        self.hr_slices = hr_slices
        self.transform = transform

    def __len__(self):
        return len(self.lr_slices)

    def __getitem__(self, idx):
        lr_slice = self.lr_slices[idx]  # shape [H, W]
        hr_slice = self.hr_slices[idx]  # shape [H, W]

        # Convert to float32
        lr_slice = lr_slice.astype(np.float32)
        hr_slice = hr_slice.astype(np.float32)

        # Add channel dimension: [1, H, W]
        lr_slice = np.expand_dims(lr_slice, axis=0)
        hr_slice = np.expand_dims(hr_slice, axis=0)

        # Convert to torch tensors
        lr_tensor = torch.from_numpy(lr_slice)
        hr_tensor = torch.from_numpy(hr_slice)

        if self.transform:
            lr_tensor = self.transform(lr_tensor)
            hr_tensor = self.transform(hr_tensor)

        return lr_tensor, hr_tensor

# def update_learning_rate(schedulers, val_loss):
#     """
#     Update the learning rate for each scheduler.
    
#     For schedulers of type ReduceLROnPlateau, use scheduler.step(val_loss),
#     otherwise, call scheduler.step() without arguments.
    
#     Parameters:
#       schedulers (list): List of learning rate scheduler objects.
#       val_loss (float): The validation loss used by ReduceLROnPlateau.
#     """
#     for scheduler in schedulers:
#         if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#             scheduler.step(val_loss)
#         else:
#             scheduler.step()

def main():


    # 1) Load pre-saved 2D slices (uncomment these lines)
    slices_lr_axial = np.load("data/lr_slices_axial.npy", allow_pickle=True)
    slices_hr_axial = np.load("data/hr_slices_axial.npy", allow_pickle=True)
    slices_lr_coronal = np.load("data/lr_slices_coronal.npy", allow_pickle=True)
    slices_hr_coronal = np.load("data/hr_slices_coronal.npy", allow_pickle=True)
    slices_lr_sagittal = np.load("data/lr_slices_sagittal.npy", allow_pickle=True)
    slices_hr_sagittal = np.load("data/hr_slices_sagittal.npy", allow_pickle=True)

    all_lr_volumes = np.concatenate((slices_lr_axial, slices_lr_coronal), axis=0)
    all_hr_volumes = np.concatenate((slices_hr_axial, slices_hr_coronal), axis=0)

    # Here, we just combine everything for a single training set.

    # 2) Create Dataset and DataLoader
    train_dataset = MRISuperResDataset(all_lr_volumes, all_hr_volumes)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 3) Initialize the model
    model = UNet2DDeep(in_channels=1, out_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) Define loss & optimizer
    criterion = nn.MSELoss()  # or L1Loss, SmoothL1Loss, etc.
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    # # Setup learning rate schedulers
    # from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
    # scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    # scheduler_step = StepLR(optimizer, step_size=10, gamma=0.1)
    # schedulers = [scheduler_plateau, scheduler_step]


    # 5) Train loop
    num_epochs = 20
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
        for lr_batch, hr_batch in loop:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
    
            optimizer.zero_grad()
            outputs = model(lr_batch)
            loss = criterion(outputs, hr_batch)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}")


    # Save model weights
    torch.save(model.state_dict(), "superres_unet_v5.pth")
    print("Model saved as superres_unet_v5.pth")

if __name__ == "__main__":
    main()
