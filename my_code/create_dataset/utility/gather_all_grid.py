import shutil
import os

source_folder = "/home/dym349/Desktop/diffusion_models/stablediffusion/scripts/analysis/fid/mscoco_2014_5k_test_image_text_retrieval/images_mscoco_2014_5k_test/Dennis_250"
destination_folder = '/home/dym349/Desktop/diffusion_models/stablediffusion/scripts/analysis/fid/mscoco_2014_5k_test_image_text_retrieval/images_mscoco_2014_5k_test/Dennis_250_grids'  # Change this to your actual destination folder path

# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List all directories in 'preliminary'
for folder_name in os.listdir(source_folder):
    if folder_name.isdigit():
        folder_path = os.path.join(source_folder, folder_name)
        grid_path = os.path.join(folder_path, 'grid-0' + folder_name + ".png")

        destination = os.path.join(destination_folder, folder_name+".png")
        shutil.copy(grid_path, destination)
        print(f'Copied {folder_name} from {grid_path} to {destination_folder}')

print("All applicable files have been copied.")
