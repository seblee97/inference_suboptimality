import torch
import typing


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


def partition_batch(batch: torch.Tensor, size: int) -> typing.Iterable[torch.Tensor]:
    """
    Partitions the given batch into minibatches of size |size|.

    :param batch: batch to partition
    :param size: size of the minibatches
    :yield: Minibatch of size |size|
    """
    for beg in range(0, len(batch), size):
        end = min(len(batch), beg + size)
        yield batch[beg:end]
