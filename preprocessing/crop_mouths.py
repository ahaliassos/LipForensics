"""
Pre-process dataset by extracting aligned cropped mouths. Adapted from https://github.com/mpc001/
Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/crop_mouth_from_video.py"""

import argparse
import os
from collections import deque

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import warp_img, apply_transform, cut_patch


DATASETS = {
    "FaceForensics++": [
        "Forensics/RealFF",
        "Forensics/Deepfakes",
        "Forensics/FaceSwap",
        "Forensics/Face2Face",
        "Forensics/NeuralTextures",
    ],
    "RealFF": ["Forensics/RealFF"],
    "Deepfakes": ["Forensics/Deepfakes"],
    "FaceSwap": ["Forensics/FaceSwap"],
    "Face2Face": ["Forensics/Face2Face"],
    "NeuralTextures": ["Forensics/NeuralTextures"],
    "FaceShifter": ["Forensics/FaceShifter"],
    "DeeperForensics": ["Forensics/DeeperForensics"],
    "CelebDF": ["CelebDF/RealCelebDF", "CelebDF/FakeCelebDF"],
    "DFDC": ["DFDC"],
}
STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-processing")
    parser.add_argument("--data-root", help="Root path of datasets", type=str, default="./data/datasets")
    parser.add_argument(
        "--dataset",
        help="Dataset to preprocess",
        type=str,
        choices=[
            "all",
            "FaceForensics++",
            "RealFF",
            "Deepfakes",
            "FaceSwap",
            "Face2Face",
            "NeuralTextures",
            "FaceShifter",
            "DeeperForensics",
            "CelebDF",
            "DFDC",
        ],
        default="ff",
    )
    parser.add_argument(
        "--compression",
        help="Video compression level for FaceForensics++",
        type=str,
        choices=["c0", "c23", "c40"],
        default="c23",
    )
    parser.add_argument("--mean-face", default="./preprocessing/20words_mean_face.npy", help="Mean face pathname")
    parser.add_argument("--crop-width", default=96, type=int, help="Width of mouth ROIs")
    parser.add_argument("--crop-height", default=96, type=int, help="Height of mouth ROIs")
    parser.add_argument("--start-idx", default=48, type=int, help="Start of landmark index for mouth")
    parser.add_argument("--stop-idx", default=68, type=int, help="End of landmark index for mouth")
    parser.add_argument("--window-margin", default=12, type=int, help="Window margin for smoothed_landmarks")

    args = parser.parse_args()
    return args


def crop_video_and_save(video_path, landmarks_dir, target_dir, mean_face_landmarks, args):
    """ "
    Align frames and crop mouths. The landmarks are smoothed over 12 frames to account for motion jitter, and each frame
    is affine warped to the mean face via five landmarks (around the eyes and nose). The mouth is cropped in each frame
    by resizing the image and then extracting a fixed 96 by 96 region centred around the mean mouth landmark.

    Parameters
    ----------
    video_path : str
        Path to video directory containing frames of faces
    landmarks_dir : str
        Path to directory of landmarks for each frame
    target_dir : str
        Path to target directory for cropped frames
    mean_face_landmarks : numpy.array
        Landmarks for the mean face of a dataset (in this case, the LRW dataset)
    args
        Further options
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    frame_names = sorted(os.listdir(video_path))

    q_frames, q_landmarks, q_name = deque(), deque(), deque()
    for i, frame_name in enumerate(frame_names):
        with Image.open(os.path.join(video_path, frame_name)) as pil_img:
            img = np.array(pil_img)
        landmarks = np.load(os.path.join(landmarks_dir, f"{frame_name[:-4]}.npy"))

        # Add elements to the queues
        q_frames.append(img)
        q_landmarks.append(landmarks)
        q_name.append(frame_name)

        if len(q_frames) == args.window_margin:  # Wait until queues are large enough
            smoothed_landmarks = np.mean(q_landmarks, axis=0)

            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frames.popleft()
            cur_name = q_name.popleft()

            # Get aligned frame as well as affine transformation that produced it
            trans_frame, trans = warp_img(
                smoothed_landmarks[STABLE_POINTS, :], mean_face_landmarks[STABLE_POINTS, :], cur_frame, STD_SIZE
            )

            # Apply that affine transform to the landmarks
            trans_landmarks = trans(cur_landmarks)

            # Crop mouth region
            cropped_frame = cut_patch(
                trans_frame,
                trans_landmarks[args.start_idx : args.stop_idx],
                args.crop_height // 2,
                args.crop_width // 2,
            )

            # Save image
            target_path = os.path.join(target_dir, cur_name)
            Image.fromarray(cropped_frame.astype(np.uint8)).save(target_path)

    # Process remaining frames in the queue
    while q_frames:
        cur_frame = q_frames.popleft()
        cur_name = q_name.popleft()
        cur_landmarks = q_landmarks.popleft()

        trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
        trans_landmarks = trans(cur_landmarks)

        cropped_frame = cut_patch(
            trans_frame, trans_landmarks[args.start_idx : args.stop_idx], args.crop_height // 2, args.crop_width // 2
        )

        target_path = os.path.join(target_dir, cur_name)
        Image.fromarray(cropped_frame.astype(np.uint8)).save(target_path)


def main():
    args = parse_args()

    mean_face_landmarks = np.load(args.mean_face)

    if args.dataset == "all":
        datasets = [
            "Forensics/RealFF",
            "Forensics/Deepfakes",
            "Forensics/FaceSwap",
            "Forensics/Face2Face",
            "Forensics/NeuralTextures",
            "Forensics/FaceShifter",
            "Forensics/DeeperForensics",
            "CelebDF/RealCelebDF",
            "CelebDF/FakeCelebDF",
            "DFDC",
        ]
    else:
        datasets = DATASETS[args.dataset]

    for dataset in datasets:
        compression = (
            args.compression if dataset not in ("CelebDF/RealCelebDF", "CelebDF/FakeCelebDF", "DFDC") else ""
        )
        root = os.path.join(args.data_root, dataset, compression)
        videos_root = os.path.join(root, "images")
        landmarks_root = os.path.join(root, "landmarks")

        video_folders = sorted(os.listdir(videos_root))

        print(f"\nProcessing {dataset}...")
        for video in tqdm(video_folders):
            target_dir = os.path.join(root, "cropped_mouths", video)
            video_dir = os.path.join(videos_root, video)
            landmarks_dir = os.path.join(landmarks_root, video)

            crop_video_and_save(video_dir, landmarks_dir, target_dir, mean_face_landmarks, args)


if __name__ == "__main__":
    main()
