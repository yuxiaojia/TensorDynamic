# baseline_gemm_bounded.py
import torch

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

device = "cuda"

LOW, HIGH = -1.0, 1.0

A = (HIGH - LOW) * torch.rand(
    100, 16, 32, 32, device=device, dtype=torch.float16
) + LOW

B = (HIGH - LOW) * torch.rand(
    100, 16, 32, 32, device=device, dtype=torch.float16
) + LOW

A = A.view(-1, 32, 32)
B = B.view(-1, 32, 32)

C = torch.bmm(A, B)
C = C.view(100, 16, 32, 32)

torch.save(C.cpu(), "gemm_out.pt")
print("GEMM done:", C.shape)
