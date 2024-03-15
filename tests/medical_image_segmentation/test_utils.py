import time
import unittest

from medical_image_segmentation.analyze_data.utils import process_files


def dummy_processing_function(file_path: str, *args, **kwargs) -> dict:
    """A dummy processing function for testing."""
    millisecond = 0.001
    time.sleep(millisecond * 10)
    return {file_path: {}}


class TestProcessFiles(unittest.TestCase):

    def test_single_process(self):
        """Test processing with a single process."""
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
        results = process_files(image_paths, dummy_processing_function)
        self.assertEqual(len(results), 3)
        for path in image_paths:
            self.assertIn(path, results)

    def test_multiple_processes(self):
        """Test processing with multiple processes."""
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
        results = process_files(image_paths, dummy_processing_function, num_processes=2)
        self.assertEqual(len(results), 3)
        for path in image_paths:
            self.assertIn(path, results)

    def test_invalid_num_processes(self):
        """Test processing with an invalid number of processes."""
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
        with self.assertRaises(ValueError):
            process_files(image_paths, dummy_processing_function, num_processes=0)

    def test_stress(self):
        """Stress test with a large number of files."""
        num_files = 10_000
        image_paths = [f"image{i}.jpg" for i in range(num_files)]
        results = process_files(image_paths, dummy_processing_function, num_processes=4)
        self.assertEqual(len(results), num_files)
        for path in image_paths:
            self.assertIn(path, results)


if __name__ == '__main__':
    unittest.main()
