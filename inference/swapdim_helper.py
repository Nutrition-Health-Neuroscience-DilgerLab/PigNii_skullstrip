import os
import subprocess

def fslswapdim_batch(input_folder, output_folder):
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder '{input_folder}' does not exist.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        
        if os.path.isfile(input_file):
            output_file = os.path.join(output_folder, filename)
            command = ["fslswapdim", input_file, "-x", "y", "z", output_file]
            
            try:
                subprocess.run(command, check=True)
                print(f"Processed {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process {filename}: {e}")

input_folder = "inference/imgs"
output_folder = "inference/ro_imgs"
fslswapdim_batch(input_folder, output_folder)