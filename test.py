import argparse

import torch
from torchvision.transforms import Compose, CenterCrop

from data.transforms import NormalizeVideo, ToTensorVideo
from models.spatiotemporal_net import get_model

import cv2
import numpy as np

def load_video_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frames = []
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Append the frame to the list
        frames.append(frame_rgb)

    # Release the video capture object
    cap.release()

    # Convert the list of frames to a NumPy array
    frames_array = np.array(frames)

    return frames_array

# Example usage:
# video_path = "path/to/your/video.mp4"
# video_frames = load_video_frames(video_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LipForensics model on a single video")
    parser.add_argument("--video_path", help="Path to the input video file", type=str, required=True)
    parser.add_argument("--weights_forgery_path", help="Path to pretrained weights for forgery detection",
                        type=str, default="./models/weights/lipforensics_ff.pth")
    parser.add_argument("--frames_per_clip", default=25, type=int)
    parser.add_argument("--device", help="Device to put tensors on", type=str, default="cuda:0")

    args = parser.parse_args()
    return args

def evaluate_video(model, video_frames, args):
    model.eval()

    transform = Compose([ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))])
    video_tensor = transform(video_frames)

    with torch.no_grad():
        video_tensor = video_tensor.unsqueeze(0).to(args.device)
        logits = model(video_tensor, lengths=[args.frames_per_clip])
        score = torch.sigmoid(logits).item()

    return score

def main():
    args = parse_args()

    # Load LipForensics model
    model = get_model(weights_forgery_path=args.weights_forgery_path)

    # Load video frames
    video_frames = load_video_frames(args.video_path)

    # Evaluate video
    score = evaluate_video(model, video_frames, args)
    print(f"Forgery score for {args.video_path}: {score}")

if __name__ == "__main__":
    main()
