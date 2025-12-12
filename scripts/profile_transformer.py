import argparse

import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from models.transformer_baseline import AETransformerBaseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--input_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--trace_dir", type=str, default="./tb_traces")
    parser.add_argument("--trace_file", type=str, default="transformer_profile.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AETransformerBaseline(
        input_dim=args.input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        num_classes=2,
    ).to(device)
    model.eval()

    x = torch.randn(args.batch_size, args.seq_len, args.input_dim, device=device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=tensorboard_trace_handler(args.trace_dir),
    ) as prof:
        for _ in range(args.steps):
            with record_function("forward"):
                with torch.no_grad():
                    _ = model(x)

    if device.type == "cuda":
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    else:
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    prof.export_chrome_trace(args.trace_file)
    print(f"Chrome trace saved to {args.trace_file}")
    print(f"TensorBoard traces under {args.trace_dir}")


if __name__ == "__main__":
    main()
