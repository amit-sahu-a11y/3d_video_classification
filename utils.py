import torch
import cv2
import numpy as np
from torchvision import transforms

def preprocess_video(video_path, transform, frames_per_clip=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < frames_per_clip and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))
        count += 1
    cap.release()

    while len(frames) < frames_per_clip:
        frames.append(frames[-1])

    video_tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)
    return video_tensor
