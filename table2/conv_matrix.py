# baseline_cnn_bounded.py
import torch
import torch.nn as nn

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

device = "cuda"

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        return self.conv(x)

model = SimpleCNN().to(device).half()
model.eval()

# Control weight range explicitly
with torch.no_grad():
    model.conv.weight.copy_(
        0.2 * torch.rand_like(model.conv.weight) - 0.1
    )

# Input range control
x = 2.0 * torch.rand(
    100, 16, 32, 32, device=device, dtype=torch.float16
) - 1.0

with torch.no_grad():
    y = model(x)

torch.save(y.cpu(), "cnn_out.pt")
print("CNN done:", y.shape)
