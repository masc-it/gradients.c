# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy"]
# ///

import torch


def main() -> None:
    torch.manual_seed(23)
    x = torch.randn(4, 7, dtype=torch.float32, requires_grad=True)
    w = torch.randn(7, 6, dtype=torch.float32, requires_grad=True)
    b = torch.randn(6, dtype=torch.float32, requires_grad=True)
    dy = torch.randn(4, 6, dtype=torch.float32)
    y = x @ w + b
    y.backward(dy)
    dx = dy @ w.detach().T
    dw = x.detach().T @ dy
    db = dy.sum(dim=0)
    dx_err = (x.grad.detach() - dx).abs().max().item()
    dw_err = (w.grad.detach() - dw).abs().max().item()
    db_err = (b.grad.detach() - db).abs().max().item()
    assert dx_err < 1e-6, dx_err
    assert dw_err < 1e-6, dw_err
    assert db_err < 1e-6, db_err
    print(f"linear bwd reference ok dx_err={dx_err:.3e} dw_err={dw_err:.3e} db_err={db_err:.3e}")


if __name__ == "__main__":
    main()
