import math
import unittest

from pcghnet.geometry import grasp_angle_error, is_successful_grasp, rotated_iou


class GeometryTest(unittest.TestCase):
    def test_identical_rectangles_have_unit_iou(self):
        grasp = [50, 40, 30, 10, 0.4]
        self.assertAlmostEqual(rotated_iou(grasp, grasp), 1.0, places=6)

    def test_disjoint_rectangles_have_zero_iou(self):
        self.assertEqual(rotated_iou([0, 0, 10, 4, 0], [30, 30, 10, 4, 0]), 0.0)

    def test_angle_error_respects_antipodal_symmetry(self):
        self.assertAlmostEqual(grasp_angle_error(0.1, 0.1 + math.pi), 0.0, places=6)

    def test_success_uses_iou_and_angle(self):
        prediction = [50, 40, 30, 10, 0.1]
        success, overlap, _ = is_successful_grasp(
            prediction, [[50, 40, 30, 10, 0.2]]
        )
        self.assertTrue(success)
        self.assertGreater(overlap, 0.25)


if __name__ == "__main__":
    unittest.main()
