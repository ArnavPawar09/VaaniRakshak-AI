import torch
import torch.nn as nn


class VaaniNet(nn.Module):
    """
    VaaniNet
    --------
    Custom CNN for real-vs-synthetic speech detection.

    Input:
        Log-Mel spectrogram
        Shape: [batch, 1, 64, time]

    Output:
        2 classes
        0 = REAL
        1 = FAKE
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(0.15),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(0.20),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Dropout2d(0.25),

            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            # Adaptive pooling means the classifier does not
            # depend on the exact number of time frames.

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.35),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x