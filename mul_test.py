import torch

from util.save_tensor import save_tensor

if __name__ == "__main__":
    a = torch.randn(512, 512, dtype=torch.float32)
    b = torch.randn(512, 512, dtype=torch.float32)
    c = a @ b
    print(a)
    print(b)
    print(c)
    save_tensor(a, "a")
    save_tensor(b, "b")
    save_tensor(c, "c")
