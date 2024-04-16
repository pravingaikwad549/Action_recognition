import cv2
import mediapipe as mp
import numpy as np
import os
import torch
import pandas as pd
from torch import nn

class pose_extraction():
    def __init__(self, parent_folder_path):
        self.parent_folder_path = parent_folder_path
        print("Parent folder path:", parent_folder_path)

    def get_data(self, results):
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        return pose
    
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def extract_data(self):    
        for item in os.listdir(self.parent_folder_path):
            item_path = os.path.join(self.parent_folder_path, item)

            if os.path.isdir(item_path):
                files = os.listdir(item_path)
                files = [file for file in files if "_c01" in file]
                print("Folder:", item_path)
                resullt = self.get_res(files_list = files, folder_path = item_path)
                resullt = torch.tensor(resullt)
                torch.save(resullt, f"{item_path}/{item}.pt")
                print("Saved file:", item + ".pt" + "at location:", os.getcwd())

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def get_res(self, files_list, folder_path):
        res = []
        mp_holistic = mp.solutions.holistic
        for file_name in files_list:
            file_path = os.path.join(folder_path, file_name)
            cap = cv2.VideoCapture(file_path)
            file_res = []
            
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                print("Processing file:", file_name)
                while True:
                    pose = []
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image, results = self.mediapipe_detection(frame, holistic)
                    pose.append(self.get_data(results))
                    file_res.append(pose)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                
            res.append(file_res)
        res = [[arr[0] for arr in f] for f in res]
        result = []
        for i in range(len(res)):
            stacked_array = np.vstack(res[i])
            result.append(stacked_array)

        resized_arrays = []
        for array in result:
            original_shape = array.shape
            if original_shape[0] > 100:
                resized_array = array[:100, :]
            else:
                resized_array = np.pad(array, ((0, 100 - original_shape[0]), (0, 0)), mode='constant')
            resized_arrays.append(resized_array)
        result = np.stack(resized_arrays)
        return result
                    

