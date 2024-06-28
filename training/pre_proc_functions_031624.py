import os
import numpy as np
import nibabel as nib
import imageio
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

def get_img_prefixes(img_dir):
    return {f.split('_mc_restore')[0] for f in os.listdir(img_dir) if f.endswith('mc_restore.nii.gz')}

def get_mask_prefixes(mask_dir):
    return {f.split('-mask')[0] for f in os.listdir(mask_dir) if f.endswith('-mask.nii.gz')}

def remove_unmatched_files(img_dir, mask_dir):
    img_prefixes = get_img_prefixes(img_dir)
    mask_prefixes = get_mask_prefixes(mask_dir)

    # Images without masks
    for prefix in img_prefixes - mask_prefixes:
        os.remove(os.path.join(img_dir, f"{prefix}_mc_restore.nii.gz"))
        print(f"Removed {prefix}_mc_restore.nii.gz from images directory")

    # Masks without images
    for prefix in mask_prefixes - img_prefixes:
        os.remove(os.path.join(mask_dir, f"{prefix}-mask.nii.gz"))
        print(f"Removed {prefix}-mask.nii.gz from masks directory")

def normalize(img):
    """Normalize image range to [0, 255]"""
    img_min = img.min()
    img_max = img.max()
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

def nii_to_png_sag(img_dir, img_out_dir, img_fixed = "_mc_restore",mask_dir=None, mask_out_dir=None, mask_fixed = '-mask'):
    # Ensure output directories exist
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    if mask_dir != None and mask_out_dir != None:
        if not os.path.exists(mask_out_dir):
            os.makedirs(mask_out_dir)
    imgfixnii = img_fixed + '.nii.gz'
    mskfixnii = mask_fixed + '.nii.gz'
    for filename in os.listdir(img_dir):
        if imgfixnii in filename:
            # Load img
            img_path = os.path.join(img_dir, filename)
            img = nib.load(img_path).get_fdata()
            img = normalize(img)  # Normalize the entire 3D image first
            prefix = filename.split(img_fixed)[0]
            print(prefix)

            if mask_dir != None and mask_out_dir != None:
                # Load mask with matching prefix
                mask_filename = prefix + mskfixnii
                mask_path = os.path.join(mask_dir, mask_filename)
                mask = nib.load(mask_path).get_fdata()

            # Initialize empty slice
            empty_slice = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            # Convert to slices and save as PNG
            for i in range(img.shape[2]):
                # Create RGB image by stacking slices
                if i == 0:
                    rgb_img = np.stack([empty_slice, img[:, :, i], img[:, :, i + 1]], axis=2)
                elif i == img.shape[2] - 1:
                    rgb_img = np.stack([img[:, :, i - 1], img[:, :, i], empty_slice], axis=2)
                else:
                    rgb_img = np.stack([img[:, :, i - 1], img[:, :, i], img[:, :, i + 1]], axis=2)

                img_out_path = os.path.join(img_out_dir, f"{prefix}_slice{str(i).zfill(3)}.png")
                imageio.imwrite(img_out_path, rgb_img)

                if mask_dir != None and mask_out_dir != None:
                    # Normalize mask slice to 0-255 and save as PNG
                    mask_slice = (mask[:, :, i] * 255).astype(np.uint8)
                    mask_out_path = os.path.join(mask_out_dir, f"{prefix}_slice{str(i).zfill(3)}_mask.png")
                    imageio.imwrite(mask_out_path, mask_slice)

                    print(f"Processed {img_out_path} and {mask_out_path}")
                else:
                    print(f"Processed {img_out_path}")

def nii_to_png_cor(img_dir, img_out_dir, img_fixed = "_mc_restore",mask_dir=None, mask_out_dir=None, mask_fixed = '-mask'):
    # Ensure output directories exist
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    if mask_dir != None and mask_out_dir != None:
        if not os.path.exists(mask_out_dir):
            os.makedirs(mask_out_dir)
    imgfixnii = img_fixed + '.nii.gz'
    mskfixnii = mask_fixed + '.nii.gz'
    for filename in os.listdir(img_dir):
        if imgfixnii in filename:
            # Load img
            img_path = os.path.join(img_dir, filename)
            img = nib.load(img_path).get_fdata()
            img = normalize(img)  # Normalize the entire 3D image first
            prefix = filename.split(img_fixed)[0]
            print(prefix)

            if mask_dir != None and mask_out_dir != None:
                # Load mask with matching prefix
                mask_filename = prefix + mskfixnii
                mask_path = os.path.join(mask_dir, mask_filename)
                mask = nib.load(mask_path).get_fdata()

            # Initialize empty slice
            empty_slice = np.zeros((img.shape[0], img.shape[2]), dtype=np.uint8)

            # Convert to slices and save as PNG
            for i in range(img.shape[1]):
                # Create RGB image by stacking slices
                if i == 0:
                    rgb_img = np.stack([empty_slice, img[:, i, :], img[:, i+1, :]], axis=2)
                elif i == img.shape[1] - 1:
                    rgb_img = np.stack([img[:, i - 1, :], img[:, i, :], empty_slice], axis=2)
                else:
                    rgb_img = np.stack([img[:, i - 1, :], img[:, i, :], img[:, i + 1, :]], axis=2)

                img_out_path = os.path.join(img_out_dir, f"{prefix}_slice{str(i).zfill(3)}.png")
                imageio.imwrite(img_out_path, rgb_img)

                if mask_dir != None and mask_out_dir != None:
                    # Normalize mask slice to 0-255 and save as PNG
                    mask_slice = (mask[:, i, :] * 255).astype(np.uint8)
                    mask_out_path = os.path.join(mask_out_dir, f"{prefix}_slice{str(i).zfill(3)}_mask.png")
                    imageio.imwrite(mask_out_path, mask_slice)

                    print(f"Processed {img_out_path} and {mask_out_path}")
                else:
                    print(f"Processed {img_out_path}")


def nii_to_png_ax(img_dir, img_out_dir, img_fixed = "_mc_restore",mask_dir=None, mask_out_dir=None, mask_fixed = '-mask'):
    # Ensure output directories exist
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    if mask_dir != None and mask_out_dir != None:
        if not os.path.exists(mask_out_dir):
            os.makedirs(mask_out_dir)
    imgfixnii = img_fixed + '.nii.gz'
    mskfixnii = mask_fixed + '.nii.gz'
    for filename in os.listdir(img_dir):
        if imgfixnii in filename:
            # Load img
            img_path = os.path.join(img_dir, filename)
            img = nib.load(img_path).get_fdata()
            img = normalize(img)  # Normalize the entire 3D image first
            prefix = filename.split(img_fixed)[0]
            print(prefix)

            if mask_dir != None and mask_out_dir != None:
                # Load mask with matching prefix
                mask_filename = prefix + mskfixnii
                mask_path = os.path.join(mask_dir, mask_filename)
                mask = nib.load(mask_path).get_fdata()

            # Initialize empty slice
            empty_slice = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)

            # Convert to slices and save as PNG
            for i in range(img.shape[0]):
                # Create RGB image by stacking slices
                if i == 0:
                    rgb_img = np.stack([empty_slice, img[i, :, :], img[i+1, :, :]], axis=2)
                elif i == img.shape[0] - 1:
                    rgb_img = np.stack([img[i - 1, :, :], img[i, :, :], empty_slice], axis=2)
                else:
                    rgb_img = np.stack([img[i - 1, :, :], img[i, :, :], img[i+1, :, :]], axis=2)

                img_out_path = os.path.join(img_out_dir, f"{prefix}_slice{str(i).zfill(3)}.png")
                imageio.imwrite(img_out_path, rgb_img)

                if mask_dir != None and mask_out_dir != None:
                    # Normalize mask slice to 0-255 and save as PNG
                    mask_slice = (mask[i, :, :] * 255).astype(np.uint8)
                    mask_out_path = os.path.join(mask_out_dir, f"{prefix}_slice{str(i).zfill(3)}_mask.png")
                    imageio.imwrite(mask_out_path, mask_slice)

                    print(f"Processed {img_out_path} and {mask_out_path}")
                else:
                    print(f"Processed {img_out_path}")

def get_pig_numbers(img_dir):
    """Extract unique pig numbers from filenames in the directory."""
    pig_numbers = set()
    for filename in os.listdir(img_dir):
        if "mc_restore.nii.gz" in filename or "mc_restore_b0.nii.gz" in filename:
            parts = filename.split('_')
            for part in parts:
                if part.isdigit():
                    pig_numbers.add(int(part))
                    break
    return list(pig_numbers)

def train_test_split(pig_numbers, test_size=0.2):
    """Split pig numbers into train and test sets."""
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(pig_numbers)
    split_index = int(len(pig_numbers) * (1 - test_size))
    train_pigs = pig_numbers[:split_index]
    test_pigs = pig_numbers[split_index:]
    return train_pigs, test_pigs

def generate_summary_table(train_pigs, test_pigs):
    """Generate a summary table for train and test pig numbers."""
    # Create a DataFrame from the train and test lists
    summary_df = pd.DataFrame({
        'Category': ['Train', 'Test'],
        'Count': [len(train_pigs), len(test_pigs)],
        'Pig Numbers': [train_pigs, test_pigs]
    })
    
    # Display the DataFrame
    return summary_df

def move_validation_files(img_dir, mask_dir, train_pigs, test_pigs):
    """Move files corresponding to test pigs to validation folders."""
    val_img_dir = os.path.join(img_dir, "../valid_img")
    val_mask_dir = os.path.join(mask_dir, "../valid_masks")
    
    # Create validation directories if they don't exist
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)

    for pig_number in test_pigs:
        for filename in os.listdir(img_dir):
            if f"Pig_{pig_number}" in filename:
                shutil.move(os.path.join(img_dir, filename), val_img_dir)
                
        for filename in os.listdir(mask_dir):
            if f"Pig_{pig_number}" in filename:
                shutil.move(os.path.join(mask_dir, filename), val_mask_dir)

def proc_img_masks(img_dir, mask_dir, out_dir, img_fixed = "_mc_restore", mask_fixed = '-mask',test_size=None):

    os.makedirs(out_dir, exist_ok=True)
    sag = os.path.join(out_dir, "./sag")
    sag_train_img = os.path.join(out_dir, "./sag/train_img")
    sag_train_masks = os.path.join(out_dir, "./sag/train_masks")

    cor = os.path.join(out_dir, "./cor")
    cor_train_img = os.path.join(out_dir, "./cor/train_img")
    cor_train_masks = os.path.join(out_dir, "./cor/train_masks")

    ax = os.path.join(out_dir, "./ax")
    ax_train_img = os.path.join(out_dir, "./ax/train_img")
    ax_train_masks = os.path.join(out_dir, "./ax/train_masks")

    folders_to_make = [sag, sag_train_img, sag_train_masks,
                       cor, cor_train_img, cor_train_masks,
                       ax, ax_train_img, ax_train_masks]
    
    folders_of_interest = [(sag_train_img, sag_train_masks),
                       (cor_train_img, cor_train_masks),
                       (ax_train_img, ax_train_masks)]
    
    for fdr in folders_to_make:
        os.makedirs(fdr, exist_ok=True)

    
    nii_to_png_sag(
        img_dir = img_dir,
        img_out_dir = sag_train_img,
        img_fixed = img_fixed,
        mask_dir = mask_dir,
        mask_out_dir = sag_train_masks,
        mask_fixed = mask_fixed
    )

    nii_to_png_cor(
        img_dir = img_dir,
        img_out_dir = cor_train_img,
        img_fixed = img_fixed,
        mask_dir = mask_dir,
        mask_out_dir = cor_train_masks,
        mask_fixed = mask_fixed
    )

    nii_to_png_ax(
        img_dir = img_dir,
        img_out_dir = ax_train_img,
        img_fixed = img_fixed,
        mask_dir = mask_dir,
        mask_out_dir = ax_train_masks,
        mask_fixed = mask_fixed
    )

    if test_size is not None:
        pig_numbers = get_pig_numbers(img_dir)
        train_pigs, test_pigs = train_test_split(pig_numbers)

        for i in folders_of_interest:
            s_img_dir = i[0]
            s_mask_dir = i[1]
            move_validation_files(s_img_dir, s_mask_dir, train_pigs, test_pigs)