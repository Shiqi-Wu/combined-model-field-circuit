import os
import numpy as np
import argparse
import re
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Reaction-diffusion model')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def read_config_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_data_with_freqs(config):
    data_dir = config['data_dir']
    pattern = re.compile(r"data-(\d+)-(\d+).RawSolution")
    for item in os.listdir(data_dir):
        full_dir_path = os.path.join(data_dir, item)
        if os.path.isdir(full_dir_path):
            match = pattern.match(item)
            if match:
                freq1, freq2 = map(int, match.groups())
                print(f"Found folder: {item}, with freq1 = {freq1}, freq2 = {freq2}")
            else:
                continue

            try:
                file_names = [os.path.join(full_dir_path, f) for f in os.listdir(full_dir_path) if f.startswith('Solution_') and f.endswith('.txt')]
                # print(file_names[0])
            except FileNotFoundError:
                print(f"Directory {full_dir_path} not found.")
                continue

            file_names.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            data_dict = {}
            if freq1 != 0 and freq2 != 0:
                for i, file_name in enumerate(file_names):
                    time, data = read_data_from_txt(file_name)
                    I_p = 200 * np.sin(2 * np.pi * freq1 * time) + 200 * np.cos(2 * np.pi * freq2 * time)
                    data_dict[i] = {'time': time, 'data': data, 'I_p': I_p}
            elif freq1 != 0 and freq2 == 0:
                for i, file_name in enumerate(file_names):
                    time, data = read_data_from_txt(file_name)
                    I_p = 300 * np.sin(2 * np.pi * freq1 * time)
                    data_dict[i] = {'time': time, 'data': data, 'I_p': I_p}
            elif freq1 == 0 and freq2 != 0:
                for i, file_name in enumerate(file_names):
                    time, data = read_data_from_txt(file_name)
                    I_p = 300 * np.cos(2 * np.pi * freq2 * time)
                    data_dict[i] = {'time': time, 'data': data, 'I_p': I_p}
            
            if not os.path.exists(config['save_dir']):
                os.makedirs(config['save_dir'], exist_ok=True)

            file_path = os.path.join(config['save_dir'], f'data_dict_{freq1}_{freq2}.npy')

            np.save(file_path, data_dict)
            print(f"Data for {freq1} and {freq2} has been saved as '{file_path}' file.")



def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        time = float(lines[0].split('=')[1])
        data = [float(line.strip()) for line in lines[1:]]
    return time, data

def read_data_with_Ip(config):
    data_dir = config['data_dir']
    pattern = re.compile(r"Ip=200.*?pi(\d+).*?pi(\d+).*?\.RawSolution")
    for item in os.listdir(data_dir):
        full_dir_path = os.path.join(data_dir, item)
        if os.path.isdir(full_dir_path):
            match = pattern.match(item)
            if match:
                freq1, freq2 = map(int, match.groups())
                print(f"Found folder: {item}, with freq1 = {freq1}, freq2 = {freq2}")
            else:
                continue

            try:
                file_names = [os.path.join(full_dir_path, f) for f in os.listdir(full_dir_path) if f.startswith('Solution_') and f.endswith('.txt')]
                # print(file_names[0])
            except FileNotFoundError:
                print(f"Directory {full_dir_path} not found.")
                continue

            file_names.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
            data_dict = {}
            if freq1 != 0 and freq2 != 0:
                for i, file_name in enumerate(file_names):
                    time, data = read_data_from_txt(file_name)
                    I_p = 200 * np.sin(2 * np.pi * freq1 * time) + 200 * np.cos(2 * np.pi * freq2 * time)
                    data_dict[i] = {'time': time, 'data': data, 'I_p': I_p}
            elif freq1 != 0 and freq2 == 0:
                for i, file_name in enumerate(file_names):
                    time, data = read_data_from_txt(file_name)
                    I_p = 300 * np.sin(2 * np.pi * freq1 * time)
                    data_dict[i] = {'time': time, 'data': data, 'I_p': I_p}
            elif freq1 == 0 and freq2 != 0:
                for i, file_name in enumerate(file_names):
                    time, data = read_data_from_txt(file_name)
                    I_p = 300 * np.cos(2 * np.pi * freq2 * time)
                    data_dict[i] = {'time': time, 'data': data, 'I_p': I_p}
            
            if not os.path.exists(config['save_dir']):
                os.makedirs(config['save_dir'], exist_ok=True)

            file_path = os.path.join(config['save_dir'], f'data_dict_{freq1}_{freq2}.npy')

            np.save(file_path, data_dict)
            print(f"Data for {freq1} and {freq2} has been saved as '{file_path}' file.")




if __name__ == '__main__':
    args = parse_args()
    config = read_config_file(args.config)
    if not os.path.exists(config['data_dir']):
        raise TypeError(f"Data directory {config['data_dir']} not found.")
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'], exist_ok=True)
    read_data_with_Ip(config)
