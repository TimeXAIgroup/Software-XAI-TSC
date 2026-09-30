import torch.nn as nn


class AudioNet(nn.Module):
    """AudioNET model from [1].

    [1] Becker, Sören / Vielhaben, Johanna / Ackermann, Marcel / Müller, Klaus-Robert /
    Lapuschkin, Sebastian / Samek, Wojciech, AudioMNIST: Exploring Explainable Artificial Intelligence
    for audio analysis on a simple benchmark, 2024. - 10.1016/j.jfranklin.2023.11.038."""

    def __init__(self, input_size=1, output_size=10):
        super(AudioNet, self).__init__()

        self.net = nn.Sequential()
        self.net.append(nn.Conv1d(input_size, 100, 3, padding="same"))
        self.net.append(nn.ReLU())
        self.net.append(nn.MaxPool1d(2))
        self.net.append(nn.Conv1d(100, 64, 3, padding="same"))
        self.net.append(nn.ReLU())
        self.net.append(nn.MaxPool1d(2))
        self.net.append(nn.Conv1d(64, 128, 3, padding="same"))
        self.net.append(nn.ReLU())
        self.net.append(nn.MaxPool1d(2))
        self.net.append(nn.Conv1d(128, 128, 3, padding="same"))
        self.net.append(nn.ReLU())
        self.net.append(nn.MaxPool1d(2))
        self.net.append(nn.Conv1d(128, 128, 3, padding="same"))
        self.net.append(nn.ReLU())
        self.net.append(nn.MaxPool1d(2))
        self.net.append(nn.Conv1d(128, 128, 3, padding="same"))
        self.net.append(nn.ReLU())
        self.net.append(nn.MaxPool1d(2))
        self.net.append(nn.Flatten())
        self.net.append(nn.Linear(125 * 128, 1024))
        self.net.append(nn.Linear(1024, 512))
        self.net.append(nn.Linear(512, output_size))

    def forward(self, x):
        return self.net(x)


class GCB2020(nn.Module):
    """
    1D CNN for time-series classification.

    Stages:
      [Conv(k=3,pad=1, 64 ch) -> Pool(stride=2)] × 5
      light Dropout(0.1) after the last 4 pools
      Global Average Pooling over time
      MLP: 64 -> 64 -> 64 -> num_classes with Dropout(0.25)

    Input shape:  (B, in_channels, T)
    Output shape: (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        conv_channels: int = 64,
        feature_dropout: float = 0.10,
        head_hidden: int = 64,
        head_dropout: float = 0.25,
        pool_type: str = "avg",
    ):
        super().__init__()
        pool = nn.AvgPool1d if pool_type == "avg" else nn.MaxPool1d

        # --- feature extractor ---
        layers = []
        ch_in = in_channels
        for i in range(5):
            # conv keeps length (padding=1 for k=3) or (p=2, k=5)
            layers.append(nn.Conv1d(ch_in, conv_channels, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(pool(kernel_size=2, stride=2))
            # add 10% dropout after pool 1
            if i >= 1:
                if feature_dropout is not None:
                    layers.append(nn.Dropout(p=feature_dropout))
            ch_in = conv_channels

        self.features = nn.Sequential(*layers)

        # GAP over time → (B, 64)
        if pool_type == "avg":
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif pool_type == "max":
            self.gap = nn.AdaptiveMaxPool1d(1)

        # --- classifier head ---
        head = []
        head.append(nn.Linear(conv_channels, head_hidden))
        head.append(nn.ReLU(inplace=True))
        if head_dropout is not None:
            head.append(nn.Dropout(p=head_dropout))
        head.append(nn.Linear(head_hidden, head_hidden))
        head.append(nn.ReLU(inplace=True))
        if head_dropout is not None:
            head.append(nn.Dropout(p=head_dropout))
        head.append(nn.Linear(head_hidden, head_hidden))
        head.append(nn.ReLU(inplace=True))
        if head_dropout is not None:
            head.append(nn.Dropout(p=head_dropout))
        head.append(nn.Linear(head_hidden, num_classes))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        x = self.features(x)  # (B, 64, T/32)
        x = self.gap(x).squeeze(-1)  # (B, 64)
        logits = self.head(x)  # (B, num_classes)
        return logits


class DimSwapWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x_btf):
        # Input: (Batch, Time, Channel)
        return self.base(x_btf.transpose(1, 2))  # -> (Batch, Channel, Time) for FCN
