import torch.nn as nn

# defined a class for image classification. A Convulation model which takes input image (3x64x64) and predicts the
# class index corresponding to a caption.
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Sequential pipeline of layers
        self.net = nn.Sequential(

            # First convolution block
            nn.Conv2d(3, 16, 3),   # Input: 3 channels i.e RGB  → Output: 16 feature maps with filter size of 3
            nn.ReLU(),             # Activation function
            nn.MaxPool2d(2),       # Downsample (reduce spatial size)

            # Second convolution block
            nn.Conv2d(16, 32, 3),  # 16 → 32 feature maps
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Flatten feature maps into a vector
            nn.Flatten(),

            # Fully connected layers
            nn.Linear(32 * 14 * 14, 128),  # Feature vector → hidden layer
            nn.ReLU(),

            # Output layer (classification)
            nn.Linear(128, num_classes)    # Predict caption class
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        return self.net(x)
