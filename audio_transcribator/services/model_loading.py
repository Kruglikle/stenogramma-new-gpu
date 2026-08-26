from contextlib import contextmanager


@contextmanager
def allow_legacy_torch_checkpoint_loading():
    """Load trusted local pyannote checkpoints created before PyTorch 2.6."""
    import torch

    original_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        yield
    finally:
        torch.load = original_load
