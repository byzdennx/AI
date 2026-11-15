import torch, torchvision
print("PyTorch   :", torch.__version__)
print("CUDA aktif:", torch.cuda.is_available())   # False = pakai CPU, tidak masalah