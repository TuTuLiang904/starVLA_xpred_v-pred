import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastwam.utils.monitoring import write_histogram_svg, write_loss_curve_svg


class TestMonitoringArtifacts(unittest.TestCase):
    def test_writes_loss_curve_svg(self):
        records = [
            {"kind": "train", "step": 1, "loss": 3.0, "loss_action": 2.0, "loss_video": 1.0},
            {"kind": "train", "step": 2, "loss": 2.0, "loss_action": 1.5, "loss_video": 0.5},
            {"kind": "eval", "step": 2, "val_loss": 2.5},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "loss_curve.svg"
            write_loss_curve_svg(records, output_path)
            content = output_path.read_text(encoding="utf-8")
        self.assertIn("Training loss", content)
        self.assertIn("train/loss", content)

    def test_writes_action_error_histogram_svg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "action_error.svg"
            write_histogram_svg(np.array([0.0, 0.1, 0.2, 0.5]), output_path, bins=4)
            content = output_path.read_text(encoding="utf-8")
        self.assertIn("Absolute action error distribution", content)
        self.assertIn("<rect", content)


if __name__ == "__main__":
    unittest.main()
