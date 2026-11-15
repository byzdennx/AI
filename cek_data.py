import os
base = 'data_catsdogs'
for split in ['training_set', 'test_set']:
    for kelas in ['cats', 'dogs']:
        path = os.path.join(base, split, kelas)
        n = len(os.listdir(path))
        print(f"{split}/{kelas}: {n} gambar")