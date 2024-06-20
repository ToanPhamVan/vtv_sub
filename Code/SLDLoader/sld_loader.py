import tqdm
import random
import os
import cv2
import numpy as np
import json
import math
import torch






class SLD:
    def __init__(self,dataset_path,n_frame=30,batch_size=1,random_seed=42) -> None:
        '''
        Sign Language Dataset Loader (Preprocessing and Data Augmentation)
        '''
        self.dataset_path = dataset_path
       
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.n_frames = n_frame
        self.last_loaded_npy = {}



        
    def get_generator(self,highlight_word="",num_data=100):
        return Generator(self.dataset_path,highlight_word,self.batch_size,self.random_seed,n_frames=self.n_frames,num_data=num_data,last_loaded_npy=self.last_loaded_npy)
        
class Generator(torch.utils.data.IterableDataset):
    def __init__(self,data_paths,highlight_word,batch_size,random_seed,n_frames,num_data,last_loaded_npy) -> None:
        
        self.data_paths = data_paths
        self.highlight_word = highlight_word
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.random_seed = random_seed
        self.num_data = num_data
        self.full_data_list = os.listdir(data_paths)
        #remove highlight word from full_data_list
        self.full_data_list.remove(highlight_word)
        self.last_loaded_npy = last_loaded_npy
        #torch.manual_seed(random_seed)
        #np.random.seed(random_seed)
    def __iter__(self):
         worker_info = torch.utils.data.get_worker_info()
         if worker_info is None:  # single-process data loading, return the full iterator
             iter_start = self.start
             iter_end = self.end
         else:  # in a worker process
             # split workload
             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = self.start + worker_id * per_worker
             iter_end = min(iter_start + per_worker, self.end)
         return iter(self.get_data(iter_start, iter_end))
    def augment_data(self,data,frame_skip,time_crop, zoom_factor, rotation_matrix, shift_values,frame_shift,out_frames = 30):

        #Time crop is 1-3, first we div the video to 7 parts, then we crop part:
        #1: 1 - 3
        #2: 2 - 4
        #3: 3 - 5

        #if frame skip < 0, add frame by one (np.repeat)
        #if frame skip > 0, remove frame so duration /2
        _data = []
        frame_skip += 1
        while len(_data) < 7:
            frame_skip -= 1
            if frame_skip < 0:
                _data = np.repeat(data, abs(frame_skip), axis=0)
            elif frame_skip > 0:
                _data = data[frame_shift::frame_skip]
                
        data = _data
        #Get the duration of the video
        duration = data.shape[0]
        crop_duration = duration // 7
        start =  crop_duration * time_crop
        end = start + crop_duration * 2
        data = data[start:end]

        #center data to 0-1
        try:
            data = data - np.min(data)
        except: 
            print(f" {data.shape}:")
            print(f"{frame_skip}: {_data.shape}")

        #makesure the data is in 0-1 in x and y axis
        data[:, :, 0] = data[:, :, 0] / np.max(data[:, :, 0])
        data[:, :, 1] = data[:, :, 1] / np.max(data[:, :, 1])


        # Zoom
        data_zoomed = data * zoom_factor

        # Rotate
        center = (np.max(data_zoomed, axis=(0, 1)) - np.min(data_zoomed, axis=(0, 1))) / 2
        data_centered = data_zoomed - center
        # Shift (move)
        data_rotated = np.dot(data_centered, rotation_matrix.T)
        data_shifted = data_rotated + center

        #shift every point
        for i in range(data_rotated.shape[1]):
            shift_value = np.random.uniform( shift_values - shift_values/5, shift_values + shift_values/5, 2)
            data_shifted[:,i] = data_shifted[:,i] + shift_value


        #Np.repeat and slice to out_frames
        if out_frames > data_shifted.shape[0]:
            data_shifted = np.repeat(data_shifted, out_frames // data_shifted.shape[0] + 1, axis=0)
        data_shifted = data_shifted[:out_frames]

        return data_shifted
    def get_augmented_data(self,data):
        rotation_angle = np.random.uniform(-.1, .1)
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                            [np.sin(rotation_angle),  np.cos(rotation_angle)]])
        zoom_factor = np.random.uniform(0.8, 1.2)
        shift_values = np.random.uniform(-0.1, 0.1, 1)
        speed = np.random.choice([ 0, 1, 2, 3, 4])
        time_crop = np.random.choice([1, 2, 3])
        frame_shift = np.random.choice([ 0,0, 1, 2, 3, 4,5])
        return self.augment_data(data, speed, time_crop, zoom_factor, rotation_matrix, shift_values,frame_shift=frame_shift, out_frames=self.n_frames)

    def get_data(self,start,end):

        #i = 0
        last_true_label = 9999
        for i in range(self.num_data):

            should_load_true_label = np.random.choice([True, False], p=[0.5, 0.5])
            if not should_load_true_label and last_true_label > self.num_data // 10:
                last_true_label = 0
                should_load_true_label = True

            if should_load_true_label:
                data_point_path = os.path.join(self.data_paths,  self.highlight_word)
                if data_point_path in self.last_loaded_npy:
                    data = self.last_loaded_npy[data_point_path]
                else:
                    data = np.load(data_point_path)
                last_true_label = 0
                y = 1
            else:
                data_point_path = os.path.join(self.data_paths, random.choice(self.full_data_list))
                if data_point_path in self.last_loaded_npy:
                    data = self.last_loaded_npy[data_point_path]
                else:
                    data = np.load(data_point_path)
                last_true_label += 1
                y = 0


            augmented_data = self.get_augmented_data(data)

            #convert to tf
            X = torch.FloatTensor(augmented_data)
            y = torch.FloatTensor([y])
            yield X, y
    def __iter__(self):
        return self()
    def __len__(self):
        return self.num_data
    def __call__(self):
        return self.get_data(0,self.num_data)