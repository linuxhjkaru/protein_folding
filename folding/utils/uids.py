import torch
import random
from typing import List
import GPUtil

def check_available_gpus():
    # Lấy danh sách các GPU
    gpus = GPUtil.getGPUs()
    
    available_gpus = []
    gpu_use = []
    for gpu in gpus:
        if gpu.memoryUtil < 0.01:  # GPU đang rảnh nếu mức sử dụng bộ nhớ dưới 10%
            available_gpus.append(gpu.id)
        gpu_use.append(gpu.memoryUsed)
    print(f"AVAILABLE_GPUS: {available_gpus}, GPU_USED: {gpu_use}")
    if not available_gpus: 
        min_value = min(gpu_use)
        min_index = gpu_use.index(min_value)
        return min_index
    else:
        return random.choice(available_gpus)


def custom_loop_annealing(md_input_sizes: float):
    if md_input_sizes > 3000000:
        return 48, 15, 5, 10
    elif md_input_sizes > 2000000:
        return 99, 15, 10, 40
    elif md_input_sizes > 1500000:
        return 150, 10, 20, 120
    else:
        return 499, 3, 30, 120

def filter_log_file(input_file_path, output_file_path, start_step: int, end_step: int, mode: str = 'w'):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    with open(output_file_path, mode) as file:
        if mode == "w":
            file.write(lines[0])
        for line in lines[1:]:  # Skip the header line
            parts = line.split(',')
            if len(parts) == 2:
                step = int(parts[0].strip())
                if start_step <= step <= end_step:
                    file.write(line)