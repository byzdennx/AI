import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # Blok 1: Conv → ReLU → MaxPool
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1→32 channel
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 → 14x14

            # Blok 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32→64 channel
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 → 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                 # 64x7x7 = 3136 → vektor
            nn.Linear(64*7*7, 128),       # fully connected
            nn.ReLU(),
            nn.Dropout(0.3),              # cegah overfitting
            nn.Linear(128, 10)            # 10 kelas (angka 0-9)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = CNN()
print(model)