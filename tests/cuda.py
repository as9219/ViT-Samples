import unittest
import torch

class TestCudaAvailability(unittest.TestCase):
    def test_cuda_is_available(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA should be available on this system")

if __name__ == '__main__':
    unittest.main()
