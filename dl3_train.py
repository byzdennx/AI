import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dl2_model import CNN

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

model = CNN().to(device)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()               # cocok untuk klasifikasi multi-kelas
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training loop ---
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss, benar, total = 0, 0, 0

    for gambar, label in train_loader:
        gambar, label = gambar.to(device), label.to(device)

        optimizer.zero_grad()          # 1. reset gradient
        output = model(gambar)         # 2. forward pass
        loss = criterion(output, label)# 3. hitung error
        loss.backward()                # 4. backpropagation
        optimizer.step()               # 5. update bobot

        total_loss += loss.item()
        benar += (output.argmax(1) == label).sum().item()
        total += label.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Akurasi: {benar/total*100:.2f}%")

torch.save(model.state_dict(), 'model_cnn.pth')
print("✅ Model tersimpan ke model_cnn.pth")