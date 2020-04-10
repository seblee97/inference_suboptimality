import torch
import unittest

from torch_operations import repeat_batch, partition_batch


class TestTorchOperations(unittest.TestCase):
    """
    Contains tests for the torch_operations.py module.
    """

    def test_repeat_batch(self):
        """
        Tests the repeat_batch() function.
        """
        tests = [(torch.Tensor([1.0]), 1, torch.Tensor([1.0])),
                 (torch.Tensor([1.0]), 3, torch.Tensor([1.0, 1.0, 1.0])),
                 (torch.Tensor([[1.0], [2.0]]), 2, torch.Tensor([[1.0], [2.0], [1.0], [2.0]])),
                 (torch.Tensor([[[1.0, 2.0]]]), 3, torch.Tensor([[[1.0, 2.0]], [[1.0, 2.0]], [[1.0, 2.0]]]))]
        for i, (batch, copies, want) in enumerate(tests):
            have = repeat_batch(batch, copies)
            same = torch.all(torch.eq(have, want))
            self.assertTrue(same, f"Test {i}: Have = {have}, Want = {want}.")

    def test_partition_batch(self):
        """
        Tests the partition_batch() function.
        """
        tests = [(torch.Tensor([1.0]), 1, [torch.Tensor([1.0])]),
                 (torch.Tensor([1.0]), 2, [torch.Tensor([1.0])]),
                 (torch.Tensor([1.0, 2.0]), 2, [torch.Tensor([1.0, 2.0])]),
                 (torch.Tensor([1.0, 2.0]), 1, [torch.Tensor([1.0]), torch.Tensor([2.0])]),
                 (torch.Tensor([1.0, 2.0, 3.0]), 2, [torch.Tensor([1.0, 2.0]), torch.Tensor([2.0])])]
        for i, (batch, size, want) in enumerate(tests):
            have = [minibatch for minibatch in partition_batch(batch, size)]
            self.assertEqual(len(have), len(want), f"Test {i}: Have = {len(have)}, Want = {len(want)}.")
            same = [torch.all(torch.eq(have[i], want[i])) for i in range(len(have))]
            self.assertTrue(same, f"Test {i}: Have = {have}, Want = {want}.")


if __name__ == '__main__':
    unittest.main()
