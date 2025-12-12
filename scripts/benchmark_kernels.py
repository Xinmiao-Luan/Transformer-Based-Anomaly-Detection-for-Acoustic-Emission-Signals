import argparse
import time

import torch
import torch.nn.functional as F

from kernels.softmax_triton import softmax_triton_2d
from models.transformer_baseline import AETransformerBaseline
from models.transformer_triton import AETransformerTriton


def benchmark_softmax_2d(n_rows, n_cols, iters=50, device="cuda"):
    x = torch.randn(n_rows, n_cols, device=device)

    # Warmup
    for _ in range(10):
        F.softmax(x, dim=-1)
        softmax_triton_2d(x)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = F.softmax(x, dim=-1)
    torch.cuda.synchronize()
    t_torch = (time.time() - t0) / iters

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = softmax_triton_2d(x)
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / iters

    return t_torch, t_triton


def benchmark_models(seq_len, batch_size=32, iters=20, device="cuda"):
    input_dim = 16
    x = torch.randn(batch_size, seq_len, input_dim, device=device)

    base = AETransformerBaseline(input_dim=input_dim).to(device).eval()
    tri = AETransformerTriton(input_dim=input_dim).to(device).eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            base(x)
            tri(x)

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            base(x)
    torch.cuda.synchronize()
    t_base = (time.time() - t0) / iters

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            tri(x)
    torch.cuda.synchronize()
    t_tri = (time.time() - t0) / iters

    return t_base, t_tri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, Triton requires a CUDA-capable GPU.")
        return

    print("=== Softmax 2D benchmarks ===")
    for n_cols in [128, 256, 512, 1024, 2048]:
        n_rows = 512
        t_torch, t_triton = benchmark_softmax_2d(n_rows, n_cols, device=device)
        print(
            f"[2D softmax] rows={n_rows}, cols={n_cols}: "
            f"torch={t_torch*1e3:.3f} ms, "
            f"triton={t_triton*1e3:.3f} ms, "
            f"speedup={t_torch/t_triton:.2f}x"
        )

    print("\n=== Model forward benchmarks ===")
    for L in [128, 256, 512, 1024]:
        t_base, t_tri = benchmark_models(L, device=device)
        print(
            f"[Model] seq_len={L}: "
            f"baseline={t_base*1e3:.3f} ms, "
            f"triton={t_tri*1e3:.3f} ms, "
            f"speedup={t_base/t_tri:.2f}x"
        )


if __name__ == "__main__":
    main()
