"""Some extra transforms for video"""

import torch


def to_tensor(clip):
    """
    Cast tensor type to float, then permute dimensions from TxHxWxC to CxTxHxW, and finally divide by 255

    Parameters
    ----------
    clip : torch.tensor
        video clip
    """
    return clip.float().permute(3, 0, 1, 2) / 255.0


def normalize(clip, mean, std):
    """
    Normalise clip by subtracting mean and dividing by standard deviation

    Parameters
    ----------
    clip : torch.tensor
        video clip
    mean : tuple
        Tuple of mean values for each channel
    std : tuple
        Tuple of standard deviation values for each channel
    """
    clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        return normalize(clip, self.mean, self.std)


class ToTensorVideo:
    def __init__(self):
        pass

    def __call__(self, clip):
        return to_tensor(clip)
