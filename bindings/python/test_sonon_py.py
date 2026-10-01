"""
Automated unit test for Sonon Python shared memory client.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sonon import SononShm


class TestSononPython(unittest.TestCase):
    def setUp(self):
        self.test_path = "/tmp/test_sonon_py.bin"
        if os.path.exists(self.test_path):
            os.remove(self.test_path)
        self.client = SononShm(path=self.test_path, sample_rate=16000.0, capacity=1024)

    def tearDown(self):
        self.client.close()
        if os.path.exists(self.test_path):
            os.remove(self.test_path)

    def test_write_read_samples(self):
        samples = [0.1 * i for i in range(256)]
        written = self.client.write_samples(samples)
        self.assertEqual(written, 256)

        read_samples = self.client.read_available_samples()
        self.assertEqual(len(read_samples), 256)
        for expected, actual in zip(samples, read_samples):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_health_and_detection(self):
        health, severity = self.client.read_health()
        self.assertEqual(health, 1.0)
        self.assertEqual(severity, 0)

        detection = self.client.read_detection()
        self.assertIsNone(detection)


if __name__ == "__main__":
    unittest.main()
