import os
import re
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import cv2
import lpips
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from PIL import Image
from dreamsim import dreamsim

from pdb import set_trace

def load_image_in_correct_format(image_path, gray=False):
    img = cv2.imread(image_path)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def cal_MSE(img1, img2):
    mse_map = (img1 - img2) ** 2
    mse = mse_map.mean()
    return mse, mse_map

def load_box(mask_image_path, expand_ratio=1):
    """
    Creates a bounding box around the mask in the given image.
    
    Args:
    - mask_image_path (str): The file path to the mask image.
    
    Returns:
    - tuple: A tuple containing the coordinates of the bounding box (x, y, width, height).
    """
    # Read the image
    mask_image_path = re.sub('_mod\d+', '', mask_image_path)

    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    mask[mask==255] = 0 #delete the edge
    y_dim, x_dim = mask.shape[0]-1, mask.shape[1]-1
    
    # Check if the image is loaded properly
    if mask is None:
        raise ValueError("Image not found or the path is incorrect.")
    
    # Find the coordinates of all non-zero pixels
    ys, xs = np.where(mask > 0)
    
    # Find the bounding box coordinates
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    
    x_width = x_max - x_min
    y_width = y_max - y_min

    x_center = int((x_min+x_max)/2)
    y_center = int((y_min+y_max)/2)

    x_min = max(0, x_center - int(x_width/2*expand_ratio))
    x_max = min(x_dim, x_center + int(x_width/2*expand_ratio))

    y_min = max(0, y_center - int(y_width/2*expand_ratio))
    y_max = min(y_dim, y_center + int(y_width/2*expand_ratio))

    # Initialize a new binary mask
    binary_mask = np.zeros_like(mask)

    # Set pixels within the bounding box to 1
    binary_mask[y_min:y_max, x_min:x_max] = 1

    return binary_mask, (y_min, y_max, x_min, x_max)

def load_contour(mask_image_path, expand_area_ratio=1):
    '''
    Load mask and dilate it to a certain multiple of its area.

    Args:
    - mask_image_path (str): The file path to the mask image.
    - expand_area_ratio (float): The ratio by which we want to dilate the area, 
      for example, 1.5 means the area will be dilated to 1.5 times its original size.
    
    Returns:
    - mask (numpy.ndarray): The dilated mask image with value of 1.
    '''

    # Load the mask image in grayscale
    mask = cv2.imread(mask_image_path, 0)
    # To zero out the edge
    mask[mask==255] = 0
    mask[mask!=0] = 1

    initial_area = np.sum(mask > 0)

    # Define the target area
    target_area = expand_area_ratio * initial_area
    target_area = min(target_area, mask.shape[0]*mask.shape[1])

    # Create a kernel (structuring element)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Perform dilation until the area of the mask has expanded to the target area
    current_area = initial_area
    while current_area < target_area:
        mask = cv2.dilate(mask, kernel, iterations=1)
        current_area = np.sum(mask > 0)
    return mask

def normalize_brightness(img1, img2):
    '''
    normalize the brightness of img1 and img2.
    '''
    img1[img1 == 0] = 1
    img2[img2 == 0] = 1

    # Calculate the sum of RGB values for each image (cv2 uses BGR format by default)
    sum_img1 = np.sum(img1, axis=2)  # Sum of BGR channels for image 1
    sum_img2 = np.sum(img2, axis=2)  # Sum of BGR channels for image 2

    # Calculate the mean of these sums
    mean_sum = (sum_img1 + sum_img2) / 2

    # Determine the scaling factor for each image
    scale_img1 = mean_sum / (sum_img1)
    scale_img2 = mean_sum / (sum_img2)

    # Apply the scaling factor to each image's pixel values
    img1 = np.clip(img1 * scale_img1[:, :, np.newaxis], 0, 255).astype(np.uint8)
    img2 = np.clip(img2 * scale_img2[:, :, np.newaxis], 0, 255).astype(np.uint8)

    return img1, img2

def calculate_similarity(args, binary_mask = None, box_bd=None):
    metric = args.metric
    isgray, norm_brightness = args.gray, args.norm_brightness
    compare_obj = args.compare_obj

    similarity_scores = []
    maps = []  # For storing individual SSIM components if SSIM is used
    ssim_range=(0, 255)
    
    # Initialize LPIPS if needed
    if metric == 'lpips':
        lpips_model = lpips.LPIPS(net='alex').to('cuda')  # Ensure you have a GPU for this
    if metric == 'dreamsim':
        model, preprocess = dreamsim(pretrained=True)

    # if args.is_latent:
    #     ptc_data = torch.load(args.root_dir)[args.prompt_idx][args.image_idx]
    #     file_names = range(len(ptc_data['pred_x0']))
    #     ssim_range = (torch.stack(ptc_data['pred_x0']).min().item(), torch.stack(ptc_data['pred_x0']).max().item())
    #     print("should check")
    # else:
    image_dir = os.path.join(args.root_dir, f"{args.prompt_idx:03d}", f"{args.image_idx:03d}", "x0_est")
    file_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])[::args.img_interval]     

    imgs = []
    for file_name in file_names:
        img_path = os.path.join(image_dir, file_name)
        img = load_image_in_correct_format(img_path, isgray)
        imgs.append(preprocess(Image.fromarray(img)).to("cuda"))

        # if args.is_latent:
        #     img1 = ptc_data['pred_x0'][i].squeeze().permute(1,2,0).cpu().numpy()

        #     if args.compare_obj == 'next':
        #         img2 = ptc_data['pred_x0'][i+1].squeeze().permute(1,2,0).cpu().numpy()
        #     # elif args.compare_obj == 'final':
        #     #     img2 = ptc_data['pred_x0'][-1].squeeze().permute(1,2,0).cpu().numpy()
        #     else:
        #         raise NotImplementedError("args.compare_obj has only two options: [next, final]")
        # img_path = os.path.join(image_dir, file_name)
        # if args.compare_obj == 'next':
        #     img2_path = os.path.join(image_dir, file_names[i+1])
        # elif args.compare_obj == 'final':
        #     img2_path = os.path.join(image_dir, file_names[-1])
        # else:
        #     raise NotImplementedError("args.compare_obj has only two options: [next, final]")

        # img_path = os.path.join(image_dir, file_name)
        # img = load_image_in_correct_format(img_path, isgray)
        
        # if norm_brightness:
        #     img1, img2 = normalize_brightness(img1, img2)

        # if args.kernal_size != 0:
        #     kernel_size = (args.kernal_size, args.kernal_size)
        #     average_kernel = np.ones(kernel_size, np.float32) / (args.kernal_size**2)
        #     img1 = cv2.filter2D(img1, -1, average_kernel)
        #     img2 = cv2.filter2D(img2, -1, average_kernel)

        # if metric in ['ssim', 'luminance', 'contrast', 'structure', 'lum_con']:
        #     if args.gray:
        #         score, map = ssim(img1, img2, data_range=ssim_range,  win_size=7, full=True, component=metric, binary_mask=binary_mask)
        #     else:
        #         score, map = ssim(img1, img2, data_range=ssim_range,  win_size=7, channel_axis=2, full=True, component=metric, binary_mask=binary_mask)
        #     similarity_scores.append(score)
        #     maps.append(map)
        # elif metric == 'psnr':
        #     psnr_value = psnr(img1, img2)
        #     similarity_scores.append(psnr_value)
        # elif metric == 'lpips':
        #     # Convert images to LPIPS format and compute score
        #     img1_tensor = lpips.im2tensor(lpips.load_image(img1_path))  # Convert to LPIPS input format
        #     img2_tensor = lpips.im2tensor(lpips.load_image(img2_path))
        #     score = lpips_model.forward(img1_tensor.to('cuda'), img2_tensor.to('cuda'))
        #     similarity_scores.append(score.item())
        
        # elif metric == 'MSE':
            # score, map = cal_MSE(img1, img2)
            # similarity_scores.append(score)
            # maps.append(map)
        
    if metric == 'dreamsim':
        # if binary_mask is not None:
        #     img1 *= binary_mask[:,:,np.newaxis]
        #     img2 *= binary_mask[:,:,np.newaxis]
            
        #     if args.reshape2mask:
        #         img1 = img1[box_bd[0]:box_bd[1],box_bd[2]:box_bd[3]]
        #         img2 = img2[box_bd[0]:box_bd[1],box_bd[2]:box_bd[3]]
        imgs = torch.cat(imgs).to("cuda")
        
        img1 = imgs[:-1,:,:,:]
        img2 = imgs[1:,:,:,:]
        
        distance = (1-model(img1, img2)).tolist()

    return distance, maps

# Function to plot the results
def plot_similarity_scores(similarity_scores, metric, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(similarity_scores, marker='o', linestyle='-')
    plt.xlabel('Time step')
    plt.ylabel('Similarity Score')
    plt.title(f'Similarity Variation using {metric.upper()}')
    plt.ylim(0.3,1)
    # plt.xticks(np.arange(0,50,5))
    plt.grid(True)
    plt.savefig(out_dir)

def plot_ssim_components(ssim_components, out_dir):
    luminance, contrast, structure = zip(*ssim_components)
    plt.figure(figsize=(10, 5))
    plt.plot(luminance, label='Luminance', marker='o')
    plt.plot(contrast, label='Contrast', marker='*')
    plt.plot(structure, label='Structure', marker='x')
    plt.xlabel('Image Pair')
    plt.ylabel('Score')
    plt.title('SSIM Components Variation')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir)
    plt.close()

def plot_maps(maps, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for i, map in enumerate(maps):
        if len(map.shape)==3:
            map = map.mean(2)
        file_name = f"{i:03}.png"
        plt.figure(figsize=(10, 8))
        plt.imshow(map, cmap='seismic')
        plt.colorbar()
        plt.title(f'Heat Map of {i}')
        plt.savefig(os.path.join(out_dir, file_name))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate image similarity.")
    parser.add_argument('--root_dir', type=str, help="Directory containing PNG images or data.ptc.") #image_dir
    parser.add_argument('--prompt_idx', type=int, help="prompt index for the chosen image")
    parser.add_argument('--image_idx', type=int, help="image index under a given prompt")

    # parser.add_argument('--is_latent', action='store_true', default=False, help="set if the input is from ptc not image")
    parser.add_argument('--out_dir', help="Directory output .png.")
    parser.add_argument('--metric', type=str, choices=['ssim', 'luminance', 'contrast', 'structure', 'lpips', 'psnr', 'MSE', 'lum_con', 'dreamsim'], help='The metric to use for comparison.')
    parser.add_argument('--gray', action='store_true', default=False)
    parser.add_argument('--kernal_size', type=int, default=0)
    parser.add_argument('--mask', type=str, choices=['none', 'box', 'contour'], default='none')
    parser.add_argument('--norm_brightness', action='store_true', default=False, help="normalize the brightness of two images.")
    parser.add_argument('--compare_obj', choices=['next', 'final'], default='next', help="compare with the next image or the final image")
    parser.add_argument('--reshape2mask', action='store_true', default=False, help="for dreamsim we shape the dim of img to fit that of the mask")
    parser.add_argument('--img_interval', type=int, default=1, help="interval for which we will compare the similarity of images")


    args = parser.parse_args()

    name = str(f'{args.prompt_idx:03d}') + "_" + f"{args.image_idx:03d}"
    mask_dir = "/home/dym349/Desktop/diffusion_models/stablediffusion/outputs/txt2img-samples/preliminary/mask"

    box_bd = None
    if args.mask == 'contour':
        binary_mask = load_contour(os.path.join(mask_dir, f"{name}_color_mask.png"), 1.3)
    elif args.mask == 'box':
        binary_mask, box_bd = load_box(os.path.join(mask_dir, f"{name}_color_mask.png"), 1.3)
    else:
        binary_mask = None

    similarity_scores, ssim_maps = calculate_similarity(args, binary_mask, box_bd)
    out_file = args.metric + "_" + name
    if args.gray:
        out_file += '_gray'
    if args.kernal_size != 0:
        out_file += f"_filter_{args.kernal_size}"

    if args.mask != 'none':
        out_file += f"_{args.mask}"
    
    if args.norm_brightness:
        out_file += "_norm_brightness"
    
    if args.compare_obj == 'final':
        out_file += "_cmprFinal"

    if args.reshape2mask == True:
        out_file += "_rshDim"

    os.makedirs(args.out_dir,exist_ok=True)
    out_dir = os.path.join(args.out_dir, out_file)

    plot_similarity_scores(similarity_scores, args.metric, out_dir+ ".png")
    
    if args.metric in ['ssim', 'luminance', 'contrast', 'structure', 'lum_con']:
        os.makedirs(out_dir, exist_ok=True)
        plot_maps(ssim_maps, out_dir)

    print(similarity_scores)