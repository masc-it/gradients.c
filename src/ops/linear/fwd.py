# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

import torch


def main() -> None:
    torch.manual_seed(17)
    x = torch.randn(4, 7, dtype=torch.float32)
    w = torch.randn(7, 6, dtype=torch.float32)
    b = torch.randn(6, dtype=torch.float32)
    y_ref = torch.nn.functional.linear(x, w.T, b)
    y = x @ w + b
    max_abs = (y_ref - y).abs().max().item()
    assert max_abs < 1e-6, max_abs
    print(f"linear fwd reference ok max_abs={max_abs:.3e} shape={tuple(y.shape)}")


if __name__ == "__main__":
    main()
