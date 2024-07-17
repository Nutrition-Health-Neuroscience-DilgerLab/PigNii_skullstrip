import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import torch.nn as nn
from tqdm.notebook import tqdm
from PIL import Image
from torchvision import transforms
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as albu
from sklearn.metrics import jaccard_score, f1_score
from segmentation_models_pytorch.losses import JaccardLoss
import pandas as pd
import pre_proc_functions_031624 as proc
import torch.nn.functional as F

def pad_to_divisible_by_32(img, padding_mode='constant'):
    """
    Pads an image tensor so its height and width are divisible by 32.
    """
    height, width = img.shape[-2:]
    pad_height = (32 - height % 32) % 32
    pad_width = (32 - width % 32) % 32
    padding = [pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2]
    return F.pad(img, padding, mode=padding_mode)

def get_preprocessing(preprocessing_fn, scale_factor=2.6667):  # 1.6mm to 0.6mm scaling
    def preprocess(img):
        # Assuming 'img' is a PyTorch tensor of shape (channels, height, width)
        # Upsample
        img = F.interpolate(img, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # Pad
        img = pad_to_divisible_by_32(img)
        # Apply any additional preprocessing, e.g., normalization
        return preprocessing_fn(img)
    return preprocess

def iou_score(output, target):
    intersection = np.logical_and(output, target).sum()
    union = np.logical_or(output, target).sum()
    iou = intersection / union
    return iou

def dice_score(output, target):
    intersection = np.logical_and(output, target).sum()
    dice = (2.0 * intersection) / (output.sum() + target.sum())
    return dice

def display(display_list, epoch=-1, save_path=None, is_inference = False):
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
    
    # Compute IoU and Dice scores
    if is_inference:
        true_mask = display_list[1].astype(np.bool)
    else:
        true_mask = display_list[1].cpu().numpy().astype(np.bool)
    pred_mask = display_list[2] > 0.5  # display_list[2] is in [0,1] range

    iou = iou_score(pred_mask, true_mask)
    dice = dice_score(pred_mask, true_mask)

    # Display the scores on the plot
    plt.text(0, -0.1, f'2D IoU: {iou:.4f}, 2D Dice: {dice:.4f}', horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap = 'gray')
    plt.show()
  

def display_overlay(display_list, epoch=-1):
  plt.close()
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  plt.subplot(1, len(display_list)+1, 3)
  plt.title("Input")
  plt.imshow(display_list[0], cmap='gray')
  plt.imshow(display_list[1], cmap='jet', alpha=0.5)
  plt.axis('off')

  plt.subplot(1, len(display_list)+1, 4)
  plt.title("Output")
  plt.imshow(display_list[0], cmap='gray')
  plt.imshow(display_list[2], cmap='jet', alpha=0.5)
  plt.axis('off')

  plt.show()

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


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
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, os.path.splitext(image_id)[0] + "_mask" + os.path.splitext(image_id)[1]) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        #print("reading data ", self.images_fps[i], self.masks_fps[i], 0)
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in [255]]#self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        #print(len(mask.squeeze()))
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
def test_model(model, epoch, sample_ex):
  x, y = sample_ex
  model = model.to('cuda')
  with torch.no_grad():
      model.eval()
      output = model(x.float().to('cuda'))
  output = output[0].squeeze()
  print(x.median())
  #mask = torch.nn.functional.normalize(output).round().to('cpu').numpy()
  mask = ((output / output.max()) > 0.05).float().to('cpu').numpy()
  #print(mask.max())

  #print((x[0] > 0.6).float())
  #x = (x > 0.6).float()
  display([x[0].permute([1,2,0]).squeeze(), y[0].squeeze(), mask], epoch)
  display_overlay([x[0].permute([1,2,0]).squeeze(), y[0].squeeze(), mask], 1)
  y = y[0].squeeze()
  y = y.bool()
  mask = np.array(mask, dtype=bool)
  overlap = y*mask # Logical AND
  union = y + mask # Logical OR
  print("IoU: ", overlap.sum()/float(union.sum()))
  return overlap.sum()/float(union.sum())

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5)
    ]
    return albu.Compose(train_transform)

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

    
    proc.nii_to_png_sag(
        img_dir = img_dir,
        img_out_dir = sag_train_img,
        img_fixed = img_fixed,
        mask_dir = mask_dir,
        mask_out_dir = sag_train_masks,
        mask_fixed = mask_fixed
    )

    proc.nii_to_png_cor(
        img_dir = img_dir,
        img_out_dir = cor_train_img,
        img_fixed = img_fixed,
        mask_dir = mask_dir,
        mask_out_dir = cor_train_masks,
        mask_fixed = mask_fixed
    )

    proc.nii_to_png_ax(
        img_dir = img_dir,
        img_out_dir = ax_train_img,
        img_fixed = img_fixed,
        mask_dir = mask_dir,
        mask_out_dir = ax_train_masks,
        mask_fixed = mask_fixed
    )

    if test_size is not None:
        pig_numbers = proc.get_pig_numbers(img_dir)
        print(pig_numbers)
        train_pigs, test_pigs = proc.train_test_split(pig_numbers)

        for i in folders_of_interest:
            s_img_dir = i[0]
            s_mask_dir = i[1]
            proc.move_validation_files(s_img_dir, s_mask_dir, train_pigs, test_pigs)
    

def train_ss(img_dir,msk_dir,split_dir,metrics_out):
    proc_img_masks(img_dir,msk_dir,out_dir=split_dir,test_size=0.2)
    x_train_dir_sag = f'{split_dir}/sag/train_img'
    y_train_dir_sag = f'{split_dir}/sag/train_masks'

    x_valid_dir_sag = f'{split_dir}/sag/valid_img'
    y_valid_dir_sag = f'{split_dir}/sag/valid_masks'

    x_train_dir_cor = f'{split_dir}/cor/train_img'
    y_train_dir_cor = f'{split_dir}/cor/train_masks'

    x_valid_dir_cor = f'{split_dir}/cor/valid_img'
    y_valid_dir_cor = f'{split_dir}/cor/valid_masks'

    x_train_dir_ax = f'{split_dir}/ax/train_img'
    y_train_dir_ax = f'{split_dir}/ax/train_masks'

    x_valid_dir_ax = f'{split_dir}/ax/valid_img'
    y_valid_dir_ax = f'{split_dir}/ax/valid_masks'

    views = ['sag', 'cor', 'ax']
    CLASSES = ['brain']
    DEVICE = 'cuda'
    models = {'Unet':{'sag':{},'cor':{},'ax':{}}}
    preprocessing_fn = {}
    lr = 0.0001

    for i, view in enumerate(views):
        # create segmentation model with pretrained encoder
        model = torch.load(f'./model_checkpoints/Unet_efficientnet-b3_{view}.pth')
        optimizer = optim.Adam(model.parameters(), lr=lr)  # Whatever optimizer you used, e.g., Adam
        checkpoint = torch.load(f'./model_checkpoints/Unet_efficientnet-b3_{view}_state.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        preprocessing_fn_i = smp.encoders.get_preprocessing_fn('efficientnet-b3', 'imagenet')
        models['Unet'][view]['model'] = model
        models['Unet'][view]['optimizer'] = optimizer
        models['Unet'][view]['start_epoch'] = 0
        models['Unet'][view]['loss'] = loss
        preprocessing_fn[view] = preprocessing_fn_i

    train_loaders = {}
    valid_loaders = {}

    batchsize = 16

    train_dataset_sag = Dataset(
        x_train_dir_sag, 
        y_train_dir_sag, 
        augmentation=None,#get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn['sag']),
        classes=CLASSES,
    )

    valid_dataset_sag = Dataset(
        x_valid_dir_sag, 
        y_valid_dir_sag, 
        augmentation=None, 
        preprocessing=get_preprocessing(preprocessing_fn['sag']),
        classes=CLASSES,
    )
    train_loader_sag = DataLoader(train_dataset_sag, batch_size=batchsize, shuffle=True, num_workers=12)
    valid_loader_sag = DataLoader(valid_dataset_sag, batch_size=batchsize, shuffle=False, num_workers=12)

    train_loaders['sag'] = train_loader_sag
    valid_loaders['sag'] = valid_loader_sag

    train_dataset_cor = Dataset(
        x_train_dir_cor, 
        y_train_dir_cor, 
        augmentation=None,#get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn['cor']),
        classes=CLASSES,
    )

    valid_dataset_cor = Dataset(
        x_valid_dir_cor, 
        y_valid_dir_cor, 
        augmentation=None, 
        preprocessing=get_preprocessing(preprocessing_fn['cor']),
        classes=CLASSES,
    )

    train_loader_cor = DataLoader(train_dataset_cor, batch_size=batchsize, shuffle=True, num_workers=12)
    valid_loader_cor = DataLoader(valid_dataset_cor, batch_size=batchsize, shuffle=False, num_workers=12)

    train_loaders['cor'] = train_loader_cor
    valid_loaders['cor'] = valid_loader_cor

    train_dataset_ax = Dataset(
        x_train_dir_ax, 
        y_train_dir_ax, 
        augmentation=None,#get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn['ax']),
        classes=CLASSES,
    )

    valid_dataset_ax = Dataset(
        x_valid_dir_ax, 
        y_valid_dir_ax, 
        augmentation=None, 
        preprocessing=get_preprocessing(preprocessing_fn['ax']),
        classes=CLASSES,
    )

    train_loader_ax = DataLoader(train_dataset_ax, batch_size=batchsize, shuffle=True, num_workers=12)
    valid_loader_ax = DataLoader(valid_dataset_ax, batch_size=batchsize, shuffle=False, num_workers=12)

    train_loaders['ax'] = train_loader_ax
    valid_loaders['ax'] = valid_loader_ax
    
    sample_ex = next(iter(train_loaders[views[0]]))
    test_model(models['Unet'][views[0]]['model'], 1, sample_ex)

    for i, view in enumerate(views):
            if view != 'sag':
                    sample_ex = next(iter(train_loaders[view]))


            train_loss_list, val_loss_list = [], []
            train_iou_list, val_iou_list = [], []
            train_dice_list, val_dice_list = [], []
            model = models['Unet'][view]['model']
            print(f'training {view} __________________ 2024 Dilger Lab random water mark')
            #print(model)
            EPOCHS = 30
            device = 'cuda'

            criterion = JaccardLoss('binary')
            optimizer = models['Unet'][view]['optimizer']
            model.to(device)

            for epoch in tqdm(range(0, EPOCHS), desc="epoch", leave=False, colour='green'):
                model.train()
                train_loss, train_iou, train_dice, train_hd = 0, 0, 0, 0
                for i, data in enumerate(tqdm(train_loaders[view], desc="training", leave=False, colour='red')):
                    img, mask = data
                    img, mask = img.to(device), mask.to(device)

                    # Run prediction
                    optimizer.zero_grad()
                    y_pred = model(img)
                    loss = criterion(y_pred, mask)
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    # Convert predictions and ground truth to binary (assuming single channel output)
                    y_pred_bin = (y_pred.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
                    mask_bin = mask.cpu().numpy().astype(np.uint8)

                    # Update training metrics
                    train_iou += jaccard_score(mask_bin.flatten(), y_pred_bin.flatten())
                    train_dice += f1_score(mask_bin.flatten(), y_pred_bin.flatten())
                    

                # Average the training metrics
                train_loss /= len(train_loaders[view])
                train_iou /= len(train_loaders[view])
                train_dice /= len(train_loaders[view])
            

                # Append training metrics
                train_loss_list.append(train_loss)
                train_iou_list.append(train_iou)
                train_dice_list.append(train_dice)


                # Validation
                model.eval()
                val_loss, val_iou, val_dice, val_hd = 0, 0, 0, 0
                for i, data in enumerate(tqdm(valid_loaders[view], desc="validation", leave=False, colour='blue')):
                    img, mask = data
                    img, mask = img.to(device), mask.to(device)
                    with torch.no_grad():
                        y_pred = model(img)
                        loss = criterion(y_pred, mask)
                        val_loss += loss.item()

                        # Convert predictions and ground truth to binary (assuming single channel output)
                        y_pred_bin = (y_pred.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
                        mask_bin = mask.cpu().numpy().astype(np.uint8)

                        # Update validation metrics
                        val_iou += jaccard_score(mask_bin.flatten(), y_pred_bin.flatten())
                        val_dice += f1_score(mask_bin.flatten(), y_pred_bin.flatten())
                    

                # Average the validation metrics
                val_loss /= len(valid_loaders[view])
                val_iou /= len(valid_loaders[view])
                val_dice /= len(valid_loaders[view])


                # Append validation metrics
                val_loss_list.append(val_loss)
                val_iou_list.append(val_iou)
                val_dice_list.append(val_dice)


                test_model(model, epoch, sample_ex)
                print(f'{epoch}_{view}, metrics at: \ntrain loss - {train_loss} \ntrain IOU - {train_iou} \ntrain dice - {train_dice} \nval loss - {val_loss} \nval IOU - {val_iou} \nval_dice - {val_dice}')
                torch.save(model, f'./transfered_checkpoints/{epoch}_{view}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    # any other metrics you might find useful
                }, f'./transfered_checkpoints/{epoch}_{view}_state.pth')
            # Create a DataFrame and save to CSV
            metrics_df = pd.DataFrame({
                'Epoch': list(range(1, EPOCHS + 1)),
                'Train_Loss': train_loss_list,
                'Validation_Loss': val_loss_list,
                'Train_IOU': train_iou_list,
                'Validation_IOU': val_iou_list,
                'Train_Dice': train_dice_list,
                'Validation_Dice': val_dice_list,

            })
            metrics_df.to_csv(f'./transfered_checkpoints/{epoch}_{view}_training_metrics.csv', index=False)