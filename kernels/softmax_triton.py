import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_rowwise_kernel(
    X_ptr,
    Y_ptr,
    n_cols,
    stride_x,
    stride_y,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel: softmax over last dimension of a row.

    Each program handles one row: X[row, :].
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    row_x_ptr = X_ptr + row * stride_x + offs
    row_y_ptr = Y_ptr + row * stride_y + offs

    mask = offs < n_cols
    x = tl.load(row_x_ptr, mask=mask, other=-float("inf"))

    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(row_y_ptr, y, mask=mask)


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def softmax_triton_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Triton-based softmax over last dimension of a 2D tensor [N, M].

    Args:
        x: [N, M] on CUDA, float32

    Returns:
        y: [N, M]
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 2, "Expected 2D tensor"

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = _next_power_of_two(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    stride_x = x.stride(0)
    stride_y = y.stride(0)

    grid = (n_rows,)
    _softmax_rowwise_kernel[grid](
        x,
        y,
        n_cols,
        stride_x,
        stride_y,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def softmax_triton_attention(scores: torch.Tensor) -> torch.Tensor:
    """
    Softmax over the last dimension for attention scores.

    scores: [B, L, L]  (B: batch, L: seq_len)
    Returns: [B, L, L]
    """
    assert scores.dim() == 3
    B, Lq, Lk = scores.shape
    x_2d = scores.reshape(-1, Lk)  # [B*Lq, Lk]
    y_2d = softmax_triton_2d(x_2d)
    return y_2d.reshape(B, Lq, Lk)
