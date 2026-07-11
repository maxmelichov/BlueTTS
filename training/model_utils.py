import torch


def register_contiguous_grad_hook(parameter: torch.nn.Parameter) -> None:
    """Keep gradients contiguous for parameters reshaped by relative attention."""
    parameter.register_hook(lambda grad: grad.contiguous())
