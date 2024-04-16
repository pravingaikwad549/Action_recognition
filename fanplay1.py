import cv2
import mediapipe as mp
import numpy as np
import os
import torch
print("Importing Libraries")
import pandas as pd
from torch import nn
import utils
print("Imported utils")
poseextraction = utils.pose_extraction(parent_folder_path = r"/home/pravin/Desktop/rough/fanply/UCF101/UCF-101")
print(poseextraction.extract_data())