import torch
import argparse, os
import numpy as np
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from pdb import set_trace

def save_inter_images(model, inter_latents:torch.tensor, sample_root):
    os.makedirs(sample_root, exist_ok=True)

    count = 0

    for latent in inter_latents:
        img = model.decode_first_stage(latent)
        img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
        img = 255. * rearrange(img[0].cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(sample_root, f"{count:03}.png"))
        count += 1

def create_image_grid(images, rows, cols, gap=0, bg_color=(255,255,255), title="Image Grid"):
    """
    Creates a grid of images with row numbers and a title.

    Parameters:
    - images: List of PIL.Image objects.
    - rows, cols: Grid dimensions.
    - gap: Space between images.
    - bg_color: Background color (R, G, B).
    - title: Grid title.

    Returns:
    - A new PIL.Image object containing the grid.
    """
    # Determine max width and height of the images
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)

    # Adjust grid height for title and row numbers
    title_height = 30  # Adjust as needed
    row_number_width = 30  # Adjust as needed
    grid_width = cols * max_width + (cols-1) * gap + row_number_width
    grid_height = rows * max_height + (rows-1) * gap + title_height

    # Create a new image with a white background
    grid_img = Image.new('RGB', (grid_width, grid_height), color=bg_color)
    draw = ImageDraw.Draw(grid_img)

    # Optional: Load a font. Adjust path and size as needed.
    # font = ImageFont.truetype("arial.ttf", 16)
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf", 20)  # Using default font
    font_num = ImageFont.truetype("/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf", 16)  # Using default font

    # Draw the title
    draw.text((grid_width//2, 10), title, fill="black", font=font_title, anchor="mm")

    # Place images in the grid and draw row numbers
    for index, img in enumerate(images):
        row = index // cols
        col = index % cols
        x = col * (max_width + gap) + row_number_width
        y = row * (max_height + gap) + title_height
        grid_img.paste(img, (x, y))
        # Draw row number for the first column or adjust as per requirement
        if col == 0:
            draw.text((10, y + max_height//2), str(row+1), fill="black", font=font_num, anchor="lm")
        if row == 0:
            draw.text((x + max_height//2, 20), str(col+1), fill="black", font=font_num, anchor="lm")

    return grid_img


