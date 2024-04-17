import torch
from model import LSTMModel
import json
import sys
import mediapipe as mp
import cv2
import numpy as np
from collections import deque

model_inference = LSTMModel(input_dim=99, hidden_dim=128, output_dim=98, num_layers=4, seq_length=100)
model_inference.load_state_dict(torch.load(r"D:\Stuff\fanplay_assgnment\github\Action_recognition\model_action_recognition.pth"))
model_inference.eval()
print("Model loaded successfully!")

file_path = r"D:\Stuff\fanplay_assgnment\github\Action_recognition\working_model\dict_lables.json"
with open(file_path, 'r') as json_file:
    dict_labels = json.load(json_file)
print("Labels loaded successfully!")

class PoseExtraction:
    def __init__(self, seq_length=100):
        self.seq_length = seq_length
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
    
    def get_data(self, results):
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        return pose
    
    def get_pose_sequence(self):
        pose_sequence_queue = deque(maxlen=self.seq_length)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = self.mediapipe_detection(frame, holistic)
                pose = self.get_data(results)
                pose_sequence_queue.append(pose)
                if len(pose_sequence_queue) == self.seq_length:
                    pose_sequence = np.vstack(pose_sequence_queue)[np.newaxis, :, :]
                    pose_sequence_tensor = torch.tensor(pose_sequence, dtype=torch.float32)
                    yield pose_sequence_tensor, image
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    
    def mediapipe_detection(self, image, holistic):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = holistic.process(image)              # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

# Example usage
pose_extractor = PoseExtraction(seq_length=100)
for pose_sequence_tensor, image in pose_extractor.get_pose_sequence():
    output = model_inference(pose_sequence_tensor)
    action_label = dict_labels[str(output.argmax(dim=1).item())]
    max_value, max_index = torch.max(output, dim=1)
    cv2.putText(image, f"Action: {action_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Confidence: {max_value.item():.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Action Recognition', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

