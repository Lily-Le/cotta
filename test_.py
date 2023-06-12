import torch
import unittest
from prompters import Prompter
import argparse
class TestPrompter(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(prompt_size=16, image_size=64)
        self.prompter = Prompter(self.args)

    def test_forward(self):
        x = torch.randn(1, 3, self.args.image_size, self.args.image_size)
        output = self.prompter(x)
        self.assertEqual(output.shape, x.shape)

if __name__ == '__main__':
    unittest.main()