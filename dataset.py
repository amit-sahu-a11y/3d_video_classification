import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, frames_per_clip=16):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.frames_per_clip = frames_per_clip

        self.video_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(classes):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for video_file in os.listdir(class_folder):
                if video_file.endswith(".avi"):  # or other formats
                    video_path = os.path.join(class_folder, video_file)
                    self.video_paths.append(video_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._load_video_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Shape: (frames_per_clip, C, H, W) -> (C, D, H, W)
        frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0

        while count < self.frames_per_clip and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            count += 1

        cap.release()

        # Pad if video is too short
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])

        return frames
