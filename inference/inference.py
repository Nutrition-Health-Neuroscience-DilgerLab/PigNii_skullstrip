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
import segmentation_models_pytorch as smp
import albumentations as albu
from PIL import Image
import re
import subprocess
import pre_proc_functions as proc
from natsort import natsorted
import pandas as pd
import shutil
import inference_flex_functions as inf

def main():
    parser = argparse.ArgumentParser(description="Inference script with specified command-line arguments.")

    # Directories
    parser.add_argument("--images_dir", 
                        type=str, 
                        default="PigNii_skullstrip-main/PigNii_skullstrip-main/example_imgs",
                        help="Path to input directory of images.")
    parser.add_argument("--study_name", 
                        type=str, 
                        default="res",
                        help="User-defined study name for outputs.")
    parser.add_argument("--metrics_out", 
                        type=str, 
                        default="res",
                        help="If set, path to store metrics, e.g. 'res.csv'.")
    parser.add_argument("--truth_folder", 
                        type=str, 
                        default="masks_swapped_val",
                        help="Folder containing ground truth masks (if you want to compute metrics).")

    # Model checkpoints
    parser.add_argument("--model_sag_path", 
                        type=str, 
                        default="PigNii_skullstrip-main/PigNii_skullstrip-main/model_checkpoints/Unet_efficientnet-b3_sag.pth",
                        help="Path to sagittal .pth checkpoint.")
    parser.add_argument("--model_cor_path", 
                        type=str, 
                        default="PigNii_skullstrip-main/PigNii_skullstrip-main/model_checkpoints/Unet_efficientnet-b3_cor.pth",
                        help="Path to coronal .pth checkpoint.")
    parser.add_argument("--model_ax_path", 
                        type=str, 
                        default="PigNii_skullstrip-main/PigNii_skullstrip-main/model_checkpoints/Unet_efficientnet-b3_ax.pth",
                        help="Path to axial .pth checkpoint.")

    # Encoder type
    parser.add_argument("--encoder_type", 
                        type=str, 
                        default="efficientnet-b3",
                        help="Encoder name for segmentation model (e.g. efficientnet-b3).")

    # Image dimensions
    parser.add_argument("--sag_dim",
                        nargs=2, 
                        type=int, 
                        default=[288, 288],
                        help="Height,Width for sagittal images.")
    parser.add_argument("--cor_dim",
                        nargs=2, 
                        type=int, 
                        default=[256, 288],
                        help="Height,Width for coronal images.")
    parser.add_argument("--ax_dim",
                        nargs=2, 
                        type=int, 
                        default=[256, 288],
                        help="Height,Width for axial images.")

    args = parser.parse_args()

    # Print them out or use them in your pipeline
    print(f"images_dir:    {args.images_dir}")
    print(f"study_name:    {args.study_name}")
    print(f"metrics_out:   {args.metrics_out}")
    print(f"truth_folder:  {args.truth_folder}")
    print()
    print(f"model_sag_path:{args.model_sag_path}")
    print(f"model_cor_path:{args.model_cor_path}")
    print(f"model_ax_path: {args.model_ax_path}")
    print()
    print(f"encoder_type:  {args.encoder_type}")
    print()
    print(f"sag_dim:       {args.sag_dim}")
    print(f"cor_dim:       {args.cor_dim}")
    print(f"ax_dim:        {args.ax_dim}")

    images_output_dir = args.study_name + "/raw_png"
    raw_png_output = args.study_name + '/raw_png_output'
    raw_npy_output = args.study_name + '/raw_npy_output'
    volumn_out = args.study_name + "/volumn_out"
    prob_out = args.study_name + "/prob_out"
    final_out = args.study_name + '/final_out'

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

    proc.proc_img_masks(img_dir=args.images_dir, out_dir=images_output_dir,img_fixed = "_mc_restore", mask_fixed = '-mask')

    files_list_sag = inf.get_files_starting_with(raw_png_sag, "Pig")
    print(sorted(files_list_sag))
    files_list_sag = sorted(files_list_sag)

    files_list_ax = inf.get_files_starting_with(raw_png_ax, "Pig")
    print(sorted(files_list_ax))
    files_list_ax = sorted(files_list_ax)

    files_list_cor = inf.get_files_starting_with(raw_png_cor, "Pig")
    print(sorted(files_list_cor))
    files_list_cor = sorted(files_list_cor)

    model_sag = torch.load(args.model_sag_path)
    model_cor = torch.load(args.model_cor_path)
    model_ax = torch.load(args.model_ax_path)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_type, "imagenet")
    sag_dataset = inf.Dataset(
        images_dir = raw_png_sag, 
        preprocessing=inf.get_preprocessing(preprocessing_fn),
        classes=['brain'],
    )

    cor_dataset = inf.Dataset(
        images_dir = raw_png_cor, 
        preprocessing=inf.get_preprocessing(preprocessing_fn),
        classes=['brain'],
    )

    ax_dataset = inf.Dataset(
        images_dir = raw_png_ax, 
        preprocessing=inf.get_preprocessing(preprocessing_fn),
        classes=['brain'],
    )

    for i in range(len(files_list_sag)):
        torch.cuda.empty_cache()
        filename = files_list_sag[i]
        x2= inf.get_data_from_filename(filename, sag_dataset)
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
        inf.display([x2[0].permute([1,2,0]).squeeze(), x2.squeeze(), mask], epoch=os.path.basename(base_name), save_path= None, is_inference=True, inference_path=(raw_png_output+'/sag'))

    for i in range(len(files_list_cor)):
        torch.cuda.empty_cache()
        filename = files_list_cor[i]
        x2= inf.get_data_from_filename(filename, cor_dataset)
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
        inf.display([x2[0].permute([1,2,0]).squeeze(), x2.squeeze(), mask], epoch=os.path.basename(base_name), save_path= None, is_inference=True, inference_path=(raw_png_output+'/cor'))

    for i in range(len(files_list_ax)):
        torch.cuda.empty_cache()
        filename = files_list_ax[i]
        x2= inf.get_data_from_filename(filename, ax_dataset)
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
        inf.display([x2[0].permute([1,2,0]).squeeze(), x2.squeeze(), mask], epoch=os.path.basename(base_name), save_path= None, is_inference=True, inference_path=(raw_png_output+'/ax'))


    input_dir_sag = f'{args.study_name}/raw_npy_output/sag'
    input_dir_cor = f'{args.study_name}/raw_npy_output/cor'
    input_dir_ax = f'{args.study_name}/raw_npy_output/ax'
    output_dir_sag = volumn_out + '/sag'
    output_dir_cor = volumn_out + '/cor'
    output_dir_ax = volumn_out + '/ax'
    source_dir = args.images_dir
    inf.stack_slices_and_save_nifti(input_dir_sag, output_dir_sag, source_dir, 2)
    inf.stack_slices_and_save_nifti(input_dir_cor, output_dir_cor, source_dir, 1)
    inf.stack_slices_and_save_nifti(input_dir_ax, output_dir_ax, source_dir, 0)

    inf.run_fslmaths(output_dir_sag, output_dir_cor, output_dir_ax, final_out)
    ###Uncomment if calculating metrics####
    #outputcsv_dice = metrics_out + f'dice.csv' 
    #outputcsv_iou = metrics_out + f'iou.csv' 
    #inf.calc3dDice(final_out,volumn_out, truth_folder, outputcsv_dice)
    #inf.calc3dIOU(final_out,volumn_out, truth_folder, outputcsv_iou)


    print("\n[INFO] Done with inference pipeline (placeholder).")

if __name__ == "__main__":
    main()