import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dl2_model import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transform), batch_size=64
)

model = CNN().to(device)
model.load_state_dict(torch.load('model_cnn.pth'))
model.eval()   # WAJIB: matikan dropout, pakai mode evaluasi

benar, total = 0, 0
with torch.no_grad():   # tidak perlu hitung gradient → hemat memori
    for gambar, label in test_loader:
        gambar, label = gambar.to(device), label.to(device)
        output = model(gambar)
        benar += (output.argmax(1) == label).sum().item()
        total += label.size(0)

print(f"Akurasi test: {benar/total*100:.2f}%")