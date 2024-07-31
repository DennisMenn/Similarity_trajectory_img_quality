import random

def select_random_prompts(input_file, output_file, subset_size):
    with open(input_file, 'r') as f:
        prompts = f.readlines()
    
    # Ensure subset size does not exceed total number of prompts
    subset_size = min(subset_size, len(prompts))
    
    # Select random subset of prompts
    random_prompts = random.sample(prompts, subset_size)
    
    # Write selected prompts to output file
    with open(output_file, 'w') as f:
        for prompt in random_prompts:
            f.write(prompt)

input_file = 'prompts.txt'
output_file = 'rand_prompts.txt'
subset_size = 680
select_random_prompts(input_file, output_file, subset_size)
