"""Classes for face forgery datasets (FaceForensics++, FaceShifter, DeeperForensics, Celeb-DF-v2, DFDC)"""

import bisect
import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class ForensicsClips(Dataset):
    """Dataset class for FaceForensics++, FaceShifter, and DeeperForensics. Supports returning only a subset of forgery
    methods in dataset"""
    def __init__(
            self,
            real_videos,
            fake_videos,
            frames_per_clip,
            fakes=('Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures'),
            compression='c23',
            grayscale=False,
            transform=None,
            max_frames_per_video=270,
    ):
        self.frames_per_clip = frames_per_clip
        self.videos_per_type = {}
        self.paths = []
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []

        ds_types = ['RealFF'] + list(fakes)  # Since we compute AUC, we need to include the Real dataset as well
        for ds_type in ds_types:

            # get list of video names
            video_paths = os.path.join('./data/datasets/Forensics', ds_type, compression, 'cropped_mouths')
            if ds_type == 'RealFF':
                videos = sorted(real_videos)
            elif ds_type == 'DeeperForensics':  # Extra processing for DeeperForensics videos due to naming differences
                videos = []
                for f in fake_videos:
                    for el in os.listdir(video_paths):
                        if el.startswith(f.split('_')[0]):
                            videos.append(el)
                videos = sorted(videos)
            else:
                videos = sorted(fake_videos)

            self.videos_per_type[ds_type] = len(videos)
            for video in videos:
                path = os.path.join(video_paths, video)
                num_frames = min(len(os.listdir(path)), max_frames_per_video)
                num_clips = num_frames // frames_per_clip
                self.clips_per_video.append(num_clips)
                self.paths.append(path)

        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]

        path = self.paths[video_idx]
        frames = sorted(os.listdir(path))

        start_idx = clip_idx * self.frames_per_clip

        end_idx = start_idx + self.frames_per_clip

        sample = []
        for idx in range(start_idx, end_idx, 1):
            with Image.open(os.path.join(path, frames[idx])) as pil_img:
                if self.grayscale:
                    pil_img = pil_img.convert("L")
                img = np.array(pil_img)
            sample.append(img)
        sample = np.stack(sample)

        return sample, video_idx

    def __getitem__(self, idx):
        sample, video_idx = self.get_clip(idx)

        label = 0 if video_idx < self.videos_per_type['RealFF'] else 1
        label = torch.from_numpy(np.array(label))
        sample = torch.from_numpy(sample).unsqueeze(-1)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, video_idx


class CelebDFClips(Dataset):
    """Dataset class for Celeb-DF-v2"""
    def __init__(
            self,
            frames_per_clip,
            grayscale=False,
            transform=None,
    ):
        self.frames_per_clip = frames_per_clip
        self.videos_per_type = {}
        self.paths = []
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []

        ds_types = ['RealCelebDF', 'FakeCelebDF']
        for ds_type in ds_types:
            video_paths = os.path.join('./data', 'datasets', 'CelebDF', ds_type, 'cropped_mouths')
            videos = sorted(os.listdir(video_paths))

            self.videos_per_type[ds_type] = len(videos)
            for video in videos:
                path = os.path.join(video_paths, video)
                num_frames = len(os.listdir(path))
                num_clips = num_frames // frames_per_clip
                self.clips_per_video.append(num_clips)
                self.paths.append(path)

        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]

        path = self.paths[video_idx]
        frames = sorted(os.listdir(path))

        start_idx = clip_idx * self.frames_per_clip

        end_idx = start_idx + self.frames_per_clip

        sample = []
        for idx in range(start_idx, end_idx, 1):
            with Image.open(os.path.join(path, frames[idx])) as pil_img:
                if self.grayscale:
                    pil_img = pil_img.convert("L")
                img = np.array(pil_img)
            sample.append(img)

        sample = np.stack(sample)

        return sample, video_idx

    def __getitem__(self, idx):
        sample, video_idx = self.get_clip(idx)

        label = 0 if video_idx < self.videos_per_type['RealCelebDF'] else 1
        label = torch.tensor(label, dtype=torch.float32)

        sample = torch.from_numpy(sample).unsqueeze(-1)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, video_idx


class DFDCClips(Dataset):
    """Dataset class for DFDC"""
    def __init__(
            self,
            frames_per_clip,
            metadata,
            grayscale=False,
            transform=None,
    ):
        self.frames_per_clip = frames_per_clip
        self.metadata = metadata
        self.paths = []
        self.grayscale = grayscale
        self.transform = transform
        self.clips_per_video = []

        video_paths = os.path.join('./data', 'datasets', 'DFDC', 'cropped_mouths')
        videos = sorted(os.listdir(video_paths))
        for video in videos:
            path = os.path.join(video_paths, video)
            num_frames = len(os.listdir(path))
            num_clips = num_frames // frames_per_clip
            self.clips_per_video.append(num_clips)
            self.paths.append(path)

        clip_lengths = torch.as_tensor(self.clips_per_video)
        self.cumulative_sizes = clip_lengths.cumsum(0).tolist()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def get_clip(self, idx):
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]

        path = self.paths[video_idx]
        video_name = path.split('/')[-1]
        frames = sorted(os.listdir(path))

        start_idx = clip_idx * self.frames_per_clip

        end_idx = start_idx + self.frames_per_clip

        sample = []
        for idx in range(start_idx, end_idx, 1):
            with Image.open(os.path.join(path, frames[idx])) as pil_img:
                if self.grayscale:
                    pil_img = pil_img.convert("L")
                img = np.array(pil_img)
            sample.append(img)

        sample = np.stack(sample)

        return sample, video_idx, video_name

    def __getitem__(self, idx):
        sample, video_idx, video_name = self.get_clip(idx)

        label = self.metadata.loc[f'{video_name}.mp4']['is_fake']
        label = torch.tensor(label, dtype=torch.float32)

        sample = torch.from_numpy(sample).unsqueeze(-1)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, video_idx
