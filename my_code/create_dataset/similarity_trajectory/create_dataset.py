import pandas as pd
import subprocess
import ast
import os
import re

from pdb import set_trace
from tqdm import tqdm

def parse_value(input):
    sel_idx = []
    if input == "Not applicable":
        return sel_idx
    else:
        pairs = input.split(";")
        for pair in pairs:
            x, y = pair.split("-")
            idx = 3*(int(x)-1)+(int(y)-1)
            sel_idx.append(idx)
    return sel_idx

def convert_str_list(input):
    """
    This function convert the output of the img_sim.py into a list.
    """
    match = re.search(r'\[.*?\]', input, re.DOTALL)
    
    if match:
        list_str = match.group(0)
        result_list = ast.literal_eval(list_str)
    else:
        raise ValueError(f"input string {input} does not contain a list.")
    
    return result_list


def exe_file(prompt_idx:int, img_idx:int, isnatural:bool, image_root_dir:str, out_dir:str, img_interval=1, metric="dreamsim"):
    """
    send in the idx of natural/unnatural image, execute the subprocess of the imgs_sim.py.

    """
   
    if isnatural: 
        out_dir = os.path.join(out_dir, "nat")
    else:
        out_dir = os.path.join(out_dir, "unat")
  
    cpr_next = subprocess.run(['python3', IMG_SIM_PY, "--img_interval", f"{img_interval}", "--metric", metric, "--out_dir", out_dir,
                               "--root_dir", image_root_dir, "--prompt_idx", f"{prompt_idx}", "--image_idx", f"{img_idx}"], capture_output=True, text=True)
    
    cpr_final = subprocess.run(['python3', IMG_SIM_PY, "--img_interval", f"{img_interval}", "--metric", metric, "--compare_obj", "final", "--out_dir", out_dir,
                                 "--root_dir", image_root_dir, "--prompt_idx", f"{prompt_idx}", "--image_idx", f"{img_idx}"], capture_output=True, text=True)

    next_out = convert_str_list(cpr_next.stdout.strip())
    final_out = convert_str_list(cpr_final.stdout.strip())
    
    return next_out, final_out


IMG_SIM_PY = "/home/dym349/Desktop/diffusion_models/Image_quality/my_code/create_dataset/similarity_trajectory/img_sim.py"

if __name__ == "__main__":

    '''variables'''
    img_interval = 1
    metric = "dreamsim"
    
    label_csv_path = '/home/dym349/Desktop/diffusion_models/Image_quality/my_code/create_dataset/raw_data/label/Dennis-complete.csv'
    image_root_dir = "/home/dym349/Desktop/diffusion_models/Image_quality/my_code/create_dataset/raw_data/imgs/Dennis/Dennis_250_imgs"
    out_dir = "test"


    """load csv parse labels into corresponding img idx"""
    label = pd.read_csv(label_csv_path).drop('Timestamp',axis=1).drop('Username',axis=1)
    label_dict = label.to_dict()
    
    data_idx = []
    temp = {}
    for key, value in label.items():
        if "Natural" in key:
            temp["natural"] = parse_value(value[0])
        else:
            temp["unnatural"] = parse_value(value[0])
            data_idx.append(temp)
            temp = {}

    """Use idx from selected images to calculate the trajectory"""
    img_id = []
    img_naturalness = []
    cpr_next = []
    cpr_final = []

    for prompt_idx, img_dict in tqdm(enumerate(data_idx), total=len(data_idx)):
        if img_dict['natural'] != []:
            for img_idx in img_dict['natural']:
                next, final = exe_file(prompt_idx= prompt_idx, img_idx=img_idx, isnatural= True, image_root_dir= image_root_dir,
                                        out_dir= out_dir, img_interval= img_interval, metric=metric)
                img_id.append(f"{prompt_idx}_{img_idx}")
                img_naturalness.append("natural")
                cpr_next.append(next)
                cpr_final.append(final)
               
        if img_dict['unnatural'] != []:
            for img_idx in img_dict['unnatural']:
                next, final = exe_file(prompt_idx= prompt_idx, img_idx=img_idx, isnatural= False, image_root_dir= image_root_dir,
                                        out_dir= out_dir, img_interval= img_interval, metric=metric)
                img_id.append(f"{prompt_idx}_{img_idx}")
                img_naturalness.append("unnatural")
                cpr_next.append(next)
                cpr_final.append(final)
        
    dataset = {"id":img_id, "naturalness":img_naturalness, "next":cpr_next, "final":cpr_final}
    dataset = pd.DataFrame(dataset).sort_values(by="naturalness")    
    dataset.to_csv(os.path.join(out_dir,"dataset.csv"))
    