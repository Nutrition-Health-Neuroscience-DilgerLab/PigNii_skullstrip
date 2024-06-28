import os
import shutil

def grab_and_move_files(src_directory, dest_directory, file_suffix="restore.nii.gz"):
    # Ensure source directory exists
    if not os.path.exists(src_directory):
        print(f"Source directory {src_directory} does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # Loop through files in the source directory
    for filename in os.listdir(src_directory):
        if filename.endswith(file_suffix):
            src_filepath = os.path.join(src_directory, filename)
            dest_filepath = os.path.join(dest_directory, filename)
            
            # Move file from source to destination
            shutil.move(src_filepath, dest_filepath)
            print(f"Moved {filename} to {dest_directory}")

if __name__ == "__main__":
    src_dir = "/home/zimu/Desktop/Code/Skull_Stripe_dataset/8week/imgs"
    dest_dir = "/home/zimu/Desktop/Code/Skull_Stripe_dataset/img_8wks_restored"

    grab_and_move_files(src_dir, dest_dir)
