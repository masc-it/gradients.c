# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

import torch


def main() -> None:
    torch.manual_seed(7)
    x = torch.randn(4, 7, dtype=torch.float32)
    w = torch.randn(7, 6, dtype=torch.float32)
    y_ref = x @ w
    y_loop = torch.zeros_like(y_ref)
    for i in range(x.shape[0]):
        for j in range(w.shape[1]):
            acc = torch.tensor(0.0, dtype=torch.float32)
            for k in range(x.shape[1]):
                acc = acc + x[i, k] * w[k, j]
            y_loop[i, j] = acc
    max_abs = (y_ref - y_loop).abs().max().item()
    assert max_abs < 1e-6, max_abs
    print(f"matmul fwd reference ok max_abs={max_abs:.3e} shape={tuple(y_ref.shape)}")


if __name__ == "__main__":
    main()
