import transformers, torch, datasets
print("transformers:", transformers.__version__)
print("torch       :", torch.__version__)
print("datasets    :", datasets.__version__)
print("CUDA        :", torch.cuda.is_available())