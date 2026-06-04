# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

import torch


def main() -> None:
    torch.manual_seed(11)
    x = torch.randn(4, 7, dtype=torch.float32, requires_grad=True)
    w = torch.randn(7, 6, dtype=torch.float32, requires_grad=True)
    dy = torch.randn(4, 6, dtype=torch.float32)
    y = x @ w
    y.backward(dy)
    dx_ref = x.grad.detach()
    dw_ref = w.grad.detach()
    dx = dy @ w.detach().T
    dw = x.detach().T @ dy
    dx_err = (dx_ref - dx).abs().max().item()
    dw_err = (dw_ref - dw).abs().max().item()
    assert dx_err < 1e-6, dx_err
    assert dw_err < 1e-6, dw_err
    print(f"matmul bwd reference ok dx_err={dx_err:.3e} dw_err={dw_err:.3e}")


if __name__ == "__main__":
    main()
