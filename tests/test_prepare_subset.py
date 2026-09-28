import io
import math
import pickle
import unittest

import torch

from tools.prepare_ga_pp_subset import (
    SplitRemoteFile,
    safe_load_prompt,
    tensor_to_grasps,
)


class MemoryPart(object):
    def __init__(self, value):
        self.value = value
        self.size = len(value)

    def read_at(self, offset, length):
        return self.value[offset : offset + length]

    def close(self):
        pass


class PrepareSubsetTest(unittest.TestCase):
    def test_string_only_pickle_is_read_without_unpickling(self):
        payload = pickle.dumps("Lift apple by its skin.", protocol=4)
        self.assertEqual(safe_load_prompt(payload), "Lift apple by its skin.")

    def test_non_string_pickle_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_load_prompt(pickle.dumps({"prompt": "unsafe shape"}, protocol=4))

    def test_split_file_reads_across_boundary(self):
        value = SplitRemoteFile([MemoryPart(b"abc"), MemoryPart(b"defg")])
        value.seek(2)
        self.assertEqual(value.read(4), b"cdef")
        value.seek(-2, io.SEEK_END)
        self.assertEqual(value.read(), b"fg")

    def test_official_tensor_format_converts_angle_and_drops_score(self):
        tensor = torch.tensor([[0.9, 10.0, 20.0, 30.0, 8.0, 45.0]])
        grasp = tensor_to_grasps(tensor)[0]
        self.assertEqual(grasp[:4], [10.0, 20.0, 30.0, 8.0])
        self.assertAlmostEqual(grasp[4], -math.pi / 4.0)


if __name__ == "__main__":
    unittest.main()
