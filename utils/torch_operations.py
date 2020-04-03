import torch


def repeat_batch(batch: torch.Tensor, copies: int) -> torch.Tensor:
    """
    Returns a Tensor containing |copies| copies of the given batch.  Unlike
    torch.repeat(), this function repeats the Tensor along the first dimension
    and preserves all other dimensions of the Tensor.

    :param batch: batch to copy
    :param copies: number of copies
    :return: New batch containing |copies| copies of |batch|.
    """
    return batch.repeat(copies, *tuple(1 for _ in batch.size()[1:]))
