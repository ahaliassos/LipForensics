"""Implementation of ResNet+MS-TCN"""

import json

import torch
import torch.nn as nn

from .resnet import ResNet, BasicBlock
from .tcn import MultibranchTemporalConvNet


def load_json(json_fp):
    with open(json_fp, "r") as f:
        json_content = json.load(f)
    return json_content


def get_model(weights_forgery_path=None, device="cuda:0"):
    """ "
    Get Resnet+MS-TCN model, optionally with pre-trained weights

    Parameters
    ----------
    weights_forgery_path : str
        Path to file with network weights
    device : str
        Device to put model on
    """
    args_loaded = load_json("./models/configs/lrw_resnet18_mstcn.json")
    relu_type = args_loaded["relu_type"]
    tcn_options = {
        "num_layers": args_loaded["tcn_num_layers"],
        "kernel_size": args_loaded["tcn_kernel_size"],
        "dropout": args_loaded["tcn_dropout"],
        "dwpw": args_loaded["tcn_dwpw"],
        "width_mult": args_loaded["tcn_width_mult"],
    }

    model = Lipreading(num_classes=1, tcn_options=tcn_options, relu_type=relu_type)

    # load weights learned during face forgery detection
    if weights_forgery_path is not None:
        checkpoint_dict = torch.load(weights_forgery_path, map_location=lambda storage, loc: storage.cuda(device))
        model.load_state_dict(checkpoint_dict["model"])
        print("Face forgery weights loaded.")
    else:
        print("Randomly initialised weights.")

    model.to(device)
    return model


def reshape_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths):
    return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options["kernel_size"]
        self.num_kernels = len(self.kernel_sizes)

        self.mb_ms_tcn = MultibranchTemporalConvNet(
            input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw
        )
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        out = self.mb_ms_tcn(x)
        out = self.consensus_func(out, lengths)
        return self.tcn_output(out)


class Lipreading(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=500, relu_type="prelu", tcn_options={}):
        super(Lipreading, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == "prelu" else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.tcn = MultiscaleMultibranchTCN(
            input_size=self.backend_out,
            num_channels=[hidden_dim * len(tcn_options["kernel_size"]) * tcn_options["width_mult"]]
            * tcn_options["num_layers"],
            num_classes=num_classes,
            tcn_options=tcn_options,
            dropout=tcn_options["dropout"],
            relu_type=relu_type,
            dwpw=tcn_options["dwpw"],
        )

    def forward(self, x, lengths):
        x = self.frontend3D(x)
        t_new = x.shape[2]
        x = reshape_tensor(x)
        x = self.trunk(x)
        x = x.view(-1, t_new, x.size(1))
        return self.tcn(x, lengths)
