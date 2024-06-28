import nibabel as nib
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.nn as nn
from tqdm.notebook import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from PIL import Image
import re
import subprocess
import pre_proc_functions_031624 as proc
from natsort import natsorted
import pandas as pd

def iou_score(output, target):
    intersection = np.logical_and(output, target).sum()
    union = np.logical_or(output, target).sum()
    iou = intersection / union
    return iou

def dice_score(output, target):
    intersection = np.logical_and(output, target).sum()
    dice = (2.0 * intersection) / (output.sum() + target.sum())
    return dice

def display(display_list, epoch=0, save_path=None, is_inference = False, inference_path = None, showinitial = False):
    if is_inference == False and showinitial== True:
        plt.close()
        plt.figure(figsize=(15, 15))

        title = ['Augmented MPRAGE', 'Ground Truth', 'Predicted Mask']

        for i in range(len(display_list)-1):
            plt.subplot(1, len(display_list)+1, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i], cmap='gray')
            plt.axis('off')

        plt.subplot(1, len(display_list)+1, 3)
        plt.title(title[2])
        plt.imshow(display_list[2], cmap='gray')
        plt.axis('off')

        plt.subplot(1, len(display_list)+1, 4)
        plt.title("Overlay")
        plt.imshow(display_list[0], cmap='gray')
        plt.imshow(display_list[2], cmap='jet', alpha=0.5)
        plt.axis('off')
    
    #save predict png
    # If display_list[2] is a PyTorch tensor, convert it to a numpy array
    if inference_path:
        if isinstance(display_list[2], torch.Tensor):
            print('tocpu.........................................')
            img_array = display_list[2].cpu().numpy()
        else:
            img_array = display_list[2]

        # Normalize the array to 0-255 if it's floating point, as OpenCV expects uint8 type for grayscale images
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            print('converting.........................................')
            img_array = np.where(img_array > 0, 1, 0).astype(np.uint8) * 255
            #img_array = (255.0 * img_array).clip(0, 255).astype(np.uint8)

        # Save the array as a PNG using OpenCV
        print(f'writing to {inference_path}/{epoch}.png')
        print(img_array.shape)
        if not cv2.imwrite(f'{inference_path}/{epoch}.png', img_array):
            print(img_array.dtype)
            print("Error saving the image!")
            im = Image.fromarray((img_array * 255).astype('uint8'))   # Assuming img_array is in float 0.0-1.0 range
            im.save(f'{inference_path}/{epoch}.png')


    # Compute IoU and Dice scores
    #if is_inference:
    #    true_mask = display_list[1].astype(np.bool)
    #else:
    #    true_mask = display_list[1].cpu().numpy().astype(np.bool)
    #pred_mask = display_list[2] > 0.5  # display_list[2] is in [0,1] range
#
    #iou = iou_score(pred_mask, true_mask)
    #dice = dice_score(pred_mask, true_mask)

    if is_inference == False and showinitial== True:

        # Display the scores on the plot
        #plt.text(0, -0.1, f'2D IoU: {iou:.4f}, 2D Dice: {dice:.4f}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        
        if showinitial:
            plt.show()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            #plt.show()

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['unlabelled', 'brain']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir=None, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, os.path.splitext(image_id)[0] + "_mask" + os.path.splitext(image_id)[1]) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        #print("reading data ", self.images_fps[i], self.masks_fps[i], 0)
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(self.masks_fps[i], 0)
        
        
        # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in [255]]#self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        #print(len(mask.squeeze()))
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image= sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image= sample['image']
            
        return image
        
    def __len__(self):
        return len(self.ids)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_data_from_filename(filename, dataset_obj):
    """Retrieve image and mask tensors from given filename."""
    
    # Find index of the filename in the dataset
    index = dataset_obj.images_fps.index(filename)
    
    # Use the __getitem__ method of the Dataset class to get the image and mask
    image = dataset_obj.__getitem__(index)
    
    # Convert the image and mask to tensors
    # (assuming they are numpy arrays; if they are already tensors, you can skip this step)
    #image = torch.tensor(image).permute(2, 0, 1).float()
    #mask = torch.tensor(mask).float()
    
    return image

def get_files_starting_with(directory, prefix):
    """Return a list of full paths of files in the directory starting with the specified prefix."""
    return [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.startswith(prefix)]

def stack_slices_and_save_nifti(input_dir, output_dir, source_dir, direction=0):
    """
    Stack 2D slice images into 3D volumes and save as NIfTI files with original header, specifying stacking direction.
    
    Args:
    - input_dir (str): Directory containing the 2D slice `.npy` files.
    - output_dir (str): Directory where the 3D NIfTI volumes will be saved.
    - source_dir (str): Directory containing the original NIfTI files for header information.
    - direction (int): Stacking direction as an integer (0=sagittal, 1=coronal, 2=axial).
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r'Pig_((?:\d+[a-zA-Z]?)+)_slice\d+\.npy')
    
    pigs = set()
    for file in os.listdir(input_dir):
        match = pattern.match(file)
        if match:
            pigs.add(match.group(1))
    
    for pig in pigs:
        source_file = None
        for file in os.listdir(source_dir):
            if f'Pig_{pig}_mc_restore.nii.gz' in file:
                source_file = os.path.join(source_dir, file)
                break
        
        if not source_file:
            print(f"No source NIfTI file found for Pig {pig}. Skipping.")
            continue
        
        source_nifti = nib.load(source_file)
        
        slices = []
        slice_files = [file for file in os.listdir(input_dir) if re.match(rf'^Pig_{pig}_slice\d+\.npy$', file)]
        if not slice_files:
            print(f"No slice files found for Pig {pig}. Skipping.")
            continue
        
        # Sort slice files naturally
        pig_files = natsorted(slice_files)
        for file_name in pig_files:
            slice_path = os.path.join(input_dir, file_name)
            slice_data = np.load(slice_path)
            if np.all(slice_data == 1):
                slice_data = np.zeros_like(slice_data)
            slices.append(slice_data)
        
        # Adjust axis based on direction input
        volume = np.stack(slices, axis=direction)
        
        new_nifti = nib.Nifti1Image(volume, affine=source_nifti.affine, header=source_nifti.header)
        output_path = os.path.join(output_dir, f'Pig_{pig}_mask.nii.gz')
        nib.save(new_nifti, output_path)
        print(f"Saved 3D NIfTI volume for Pig {pig} to {output_path}")

def run_fslmaths(dir_sag, dir_cor, dir_ax, output_dir):
    """
    Combines masks from three directories using fslmaths and saves the output.
    
    Args:
    - dir_sag (str): Directory containing sagittal masks.
    - dir_cor (str): Directory containing coronal masks.
    - dir_ax (str): Directory containing axial masks.
    - output_dir (str): Directory where the output masks will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile a regex pattern to extract the Pig_XXXX identifier
    pattern = re.compile(r'Pig_((?:\d+[a-zA-Z]?)+)_mask.nii.gz')
    
    # Gather all unique Pig identifiers across the three directories
    pigs = set()
    for dir_path in [dir_sag, dir_cor, dir_ax]:
        for filename in os.listdir(dir_path):
            match = pattern.match(filename)
            if match:
                pigs.add(match.group(1))
    
    # For each unique Pig identifier, construct and run the fslmaths command
    for pig in pigs:
        masks = [os.path.join(dir_path, f'Pig_{pig}_mask.nii.gz') for dir_path in [dir_sag, dir_cor, dir_ax]]
        output_path = os.path.join(output_dir, f'Pig_{pig}_mask.nii.gz')
        
        # Construct the fslmaths command
        cmd = ["fslmaths", masks[0], "-add", masks[1], "-add", masks[2], "-thr", "1.5", "-bin", output_path]
        
        # Execute the command
        subprocess.run(cmd, check=True)
        print(f"Processed and saved: {output_path}")

def dice_coefficient(prediction, truth):
    """Compute the Dice coefficient."""
    intersection = (prediction * truth).sum()
    return (2. * intersection) / (prediction.sum() + truth.sum())

def calc3dDice(finalprediction_folder, directionalprediction_folder, truth_folder, output_csv):
    subjects = []
    dice_scores_final = []
    dice_scores_ax = []
    dice_scores_sag = []
    dice_scores_cor = []

    for file in os.listdir(finalprediction_folder):
        subject_id = file.split('_')[1]
        subjects.append(subject_id)

        # Load final prediction and truth
        final_pred_nii = nib.load(os.path.join(finalprediction_folder, file))
        truth_nii = nib.load(os.path.join(truth_folder, f"Pig_{subject_id}-mask.nii.gz"))

        # Compute Dice for final prediction
        dice_final = dice_coefficient(final_pred_nii.get_fdata(), truth_nii.get_fdata())
        dice_scores_final.append(dice_final)

        # Compute Dice for directional predictions
        for direction, dice_scores in zip(['ax', 'sag', 'cor'], [dice_scores_ax, dice_scores_sag, dice_scores_cor]):
            directional_pred_nii = nib.load(os.path.join(directionalprediction_folder, direction, f"Pig_{subject_id}_mask.nii.gz"))
            dice_directional = dice_coefficient(directional_pred_nii.get_fdata(), truth_nii.get_fdata())
            dice_scores.append(dice_directional)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame({
        'Subject_ID': subjects,
        'Dice_Final': dice_scores_final,
        'Dice_Ax': dice_scores_ax,
        'Dice_Sag': dice_scores_sag,
        'Dice_Cor': dice_scores_cor
    })
    df.to_csv(output_csv, index=False)

def iou_score(prediction, truth):
    """Compute the Intersection over Union (IoU) score."""
    intersection = (prediction * truth).sum()
    union = prediction.sum() + truth.sum() - intersection
    return intersection / union

def calc3dIOU(finalprediction_folder, directionalprediction_folder, truth_folder, output_csv):
    subjects = []
    iou_scores_final = []
    iou_scores_ax = []
    iou_scores_sag = []
    iou_scores_cor = []

    for file in os.listdir(finalprediction_folder):
        subject_id = file.split('_')[1]
        subjects.append(subject_id)

        # Load final prediction and truth
        final_pred_nii = nib.load(os.path.join(finalprediction_folder, file))
        truth_nii = nib.load(os.path.join(truth_folder, f"Pig_{subject_id}-mask.nii.gz"))

        # Compute IoU for final prediction
        iou_final = iou_score(final_pred_nii.get_fdata(), truth_nii.get_fdata())
        iou_scores_final.append(iou_final)

        # Compute IoU for directional predictions
        for direction, iou_scores in zip(['ax', 'sag', 'cor'], [iou_scores_ax, iou_scores_sag, iou_scores_cor]):
            directional_pred_nii = nib.load(os.path.join(directionalprediction_folder, direction, f"Pig_{subject_id}_mask.nii.gz"))
            iou_directional = iou_score(directional_pred_nii.get_fdata(), truth_nii.get_fdata())
            iou_scores.append(iou_directional)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame({
        'Subject_ID': subjects,
        'IoU_Final': iou_scores_final,
        'IoU_Ax': iou_scores_ax,
        'IoU_Sag': iou_scores_sag,
        'IoU_Cor': iou_scores_cor
    })
    df.to_csv(output_csv, index=False)

def inference(model_sag_pth, model_cor_pth, model_ax_pth, study_name, images_dir):
    images_output_dir = study_name + "/raw_png"
    raw_png_output = study_name + '/raw_png_output'
    raw_npy_output = study_name + '/raw_npy_output'
    volumn_out = study_name + "/volumn_out"
    prob_out = study_name + "/prob_out"
    final_out = study_name + '/final_out'

    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
        os.makedirs(os.path.join(images_output_dir,'sag'))
        os.makedirs(os.path.join(images_output_dir,'ax'))
        os.makedirs(os.path.join(images_output_dir,'cor'))
    raw_png_sag = os.path.join(images_output_dir,'sag')
    raw_png_ax = os.path.join(images_output_dir,'ax')
    raw_png_cor = os.path.join(images_output_dir,'cor')

    if not os.path.exists(raw_png_output):
        os.makedirs(raw_png_output)
        os.makedirs(os.path.join(raw_png_output,'sag'))
        os.makedirs(os.path.join(raw_png_output,'ax'))
        os.makedirs(os.path.join(raw_png_output,'cor'))

    raw_png_sag_output = os.path.join(raw_png_output,'sag')
    raw_png_ax_output = os.path.join(raw_png_output,'ax')
    raw_png_cor_output = os.path.join(raw_png_output,'cor')

    if not os.path.exists(raw_npy_output):
        os.makedirs(raw_npy_output)
        os.makedirs(os.path.join(raw_npy_output,'sag'))
        os.makedirs(os.path.join(raw_npy_output,'ax'))
        os.makedirs(os.path.join(raw_npy_output,'cor'))

    raw_npy_sag_output = os.path.join(raw_npy_output,'sag')
    raw_npy_ax_output = os.path.join(raw_npy_output,'ax')
    raw_npy_cor_output = os.path.join(raw_npy_output,'cor')

    if not os.path.exists(volumn_out):
        os.makedirs(volumn_out)
        os.makedirs(os.path.join(volumn_out,'sag'))
        os.makedirs(os.path.join(volumn_out,'ax'))
        os.makedirs(os.path.join(volumn_out,'cor'))

    volumn_out_sag = os.path.join(volumn_out,'sag')
    volumn_out_ax = os.path.join(volumn_out,'ax')
    volumn_out_cor = os.path.join(volumn_out,'cor')

    if not os.path.exists(prob_out):
        os.makedirs(prob_out)
        os.makedirs(os.path.join(prob_out,'sag'))
        os.makedirs(os.path.join(prob_out,'ax'))
        os.makedirs(os.path.join(prob_out,'cor'))

    prob_out_sag = os.path.join(prob_out,'sag')
    prob_out_ax = os.path.join(prob_out,'ax')
    prob_out_cor = os.path.join(prob_out,'cor')

    proc.proc_img_masks(img_dir=images_dir, out_dir=images_output_dir)

    files_list_sag = get_files_starting_with(raw_png_sag, "Pig")
    print(sorted(files_list_sag))
    files_list_sag = sorted(files_list_sag)

    files_list_ax = get_files_starting_with(raw_png_ax, "Pig")
    print(sorted(files_list_ax))
    files_list_ax = sorted(files_list_ax)

    files_list_cor = get_files_starting_with(raw_png_cor, "Pig")
    print(sorted(files_list_cor))
    files_list_cor = sorted(files_list_cor)

    model_sag = torch.load(model_sag_pth)
    model_cor = torch.load(model_cor_pth)
    model_ax = torch.load(model_ax_pth)

    preprocessing_fn = smp.encoders.get_preprocessing_fn("efficientnet-b3", "imagenet")
    sag_dataset = Dataset(
        images_dir = raw_png_sag, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=['brain'],
    )

    cor_dataset = Dataset(
        images_dir = raw_png_cor, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=['brain'],
    )

    ax_dataset = Dataset(
        images_dir = raw_png_ax, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=['brain'],
    )

    for i in range(len(files_list_sag)):
        torch.cuda.empty_cache()
        filename = files_list_sag[i]
        x2= get_data_from_filename(filename, sag_dataset)
        x2 = torch.tensor(x2).unsqueeze(0).repeat(32,1,1,1)
        model_sag = model_sag.to('cuda')
        with torch.no_grad():
            model_sag.eval()
            output = model_sag(x2.float().to('cuda'))
        output = output[0].squeeze()
        mask = ((output / output.max()) > 0.05).float().to('cpu').numpy()
        probability = ((output - output.min()) / (output.max() - output.min())).float().to('cpu').numpy()
        print("slice num : ", i)
        base_name, ext = os.path.splitext(files_list_sag[i])
        b_base = os.path.basename(base_name)
        prob_path = os.path.join(prob_out_sag, f"{b_base}.npy")
        mask_path = os.path.join(raw_npy_sag_output, f"{b_base}.npy")
        np.save(prob_path, probability)
        np.save(mask_path, mask)
        print(f'this is base_name: {files_list_sag[i]}')
        display([x2[0].permute([1,2,0]).squeeze(), x2.squeeze(), mask], epoch=os.path.basename(base_name), save_path= None, is_inference=True, inference_path=(raw_png_output+'/sag'))

    for i in range(len(files_list_cor)):
        torch.cuda.empty_cache()
        filename = files_list_cor[i]
        x2= get_data_from_filename(filename, cor_dataset)
        x2 = torch.tensor(x2).unsqueeze(0).repeat(32,1,1,1)
        model_cor = model_cor.to('cuda')
        with torch.no_grad():
            model_cor.eval()
            output = model_cor(x2.float().to('cuda'))
        output = output[0].squeeze()
        mask = ((output / output.max()) > 0.05).float().to('cpu').numpy()
        probability = ((output - output.min()) / (output.max() - output.min())).float().to('cpu').numpy()
        print("slice num : ", i)
        base_name, ext = os.path.splitext(files_list_cor[i])
        b_base = os.path.basename(base_name)
        prob_path = os.path.join(prob_out_cor, f"{b_base}.npy")
        mask_path = os.path.join(raw_npy_cor_output, f"{b_base}.npy")
        np.save(prob_path, probability)
        np.save(mask_path, mask)
        print(f'this is base_name: {files_list_cor[i]}')
        display([x2[0].permute([1,2,0]).squeeze(), x2.squeeze(), mask], epoch=os.path.basename(base_name), save_path= None, is_inference=True, inference_path=(raw_png_output+'/cor'))

    for i in range(len(files_list_ax)):
        torch.cuda.empty_cache()
        filename = files_list_ax[i]
        x2= get_data_from_filename(filename, ax_dataset)
        x2 = torch.tensor(x2).unsqueeze(0).repeat(32,1,1,1)
        model_ax = model_ax.to('cuda')
        with torch.no_grad():
            model_ax.eval()
            output = model_ax(x2.float().to('cuda'))
        output = output[0].squeeze()
        mask = ((output / output.max()) > 0.05).float().to('cpu').numpy()
        probability = ((output - output.min()) / (output.max() - output.min())).float().to('cpu').numpy()
        print("slice num : ", i)
        base_name, ext = os.path.splitext(files_list_ax[i])
        b_base = os.path.basename(base_name)
        prob_path = os.path.join(prob_out_ax, f"{b_base}.npy")
        mask_path = os.path.join(raw_npy_ax_output, f"{b_base}.npy")
        np.save(prob_path, probability)
        np.save(mask_path, mask)
        print(f'this is base_name: {files_list_ax[i]}')
        display([x2[0].permute([1,2,0]).squeeze(), x2.squeeze(), mask], epoch=os.path.basename(base_name), save_path= None, is_inference=True, inference_path=(raw_png_output+'/ax'))

    input_dir_sag = raw_npy_output + '/sag'
    input_dir_cor = raw_npy_output + '/cor'
    input_dir_ax = raw_npy_output + '/ax'
    output_dir_sag = volumn_out + '/sag'
    output_dir_cor = volumn_out + '/cor'
    output_dir_ax = volumn_out + '/ax'
    source_dir = images_dir
    # sag = 2 cor = 1 ax = 0
    stack_slices_and_save_nifti(input_dir_sag, output_dir_sag, source_dir, 2)
    stack_slices_and_save_nifti(input_dir_cor, output_dir_cor, source_dir, 1)
    stack_slices_and_save_nifti(input_dir_ax, output_dir_ax, source_dir, 0)

    run_fslmaths(output_dir_sag, output_dir_cor, output_dir_ax, final_out)