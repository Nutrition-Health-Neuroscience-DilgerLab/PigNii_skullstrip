#!/bin/bash

# Set the directory containing your data
dataDir="/home/zimu/Desktop/Code/PigNii_skull_stripe/inference/ro_imgs"

# Loop over each .nii file in the directory
for file in $dataDir/*.nii.gz
do
    # Extract the base name of the file without extension
    baseName=$(basename "$file" .nii.gz)
    echo $baseName

    # Perform motion correction
    ~/fsl/bin/mcflirt -in $file -out ${dataDir}/${baseName}_mc

    # Perform INU correction
    ~/fsl/bin/fast -B -v ${dataDir}/${baseName}_mc

done