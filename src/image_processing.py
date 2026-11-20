from PIL import Image
import numpy as np
import torch
def preprocess(path: str):
    img = Image.open(path)
    img = img.resize((28,28))
    img = img.convert("L")

    arr = np.array(img)
    tensor = torch.from_numpy(arr)
    tensor = tensor.float() / 255.0
    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)
    return tensor