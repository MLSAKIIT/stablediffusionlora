import unittest
import os
from evaluate_model import calculate_fid, calculate_lpips  # Import the functions from evaluate_model

class TestImageMetrics(unittest.TestCase):

    def setUp(self):
        """Set up test data and paths."""
        # Set paths for real and generated test images.
        # Make sure you have test images placed in these paths.
        self.real_image_path = "./real_image.jpg"
        self.generated_image_path = "./generated_image.jpg"

        # Ensure that the test images exist in the specified paths.
        self.assertTrue(os.path.exists(self.real_image_path), "Real image file not found!")
        self.assertTrue(os.path.exists(self.generated_image_path), "Generated image file not found!")

    def test_fid_score(self):
        """Test the calculation of the FID score."""
        try:
            fid_score = calculate_fid(self.real_image_path, self.generated_image_path)
            self.assertIsInstance(fid_score, float, "FID score should be a float.")
        except Exception as e:
            self.fail(f"calculate_fid() raised an exception unexpectedly: {e}")

    def test_lpips_score(self):
        """Test the calculation of the LPIPS score."""
        try:
            lpips_score = calculate_lpips(self.real_image_path, self.generated_image_path)
            self.assertIsInstance(lpips_score.item(), float, "LPIPS score should be a float.")
        except Exception as e:
            self.fail(f"calculate_lpips() raised an exception unexpectedly: {e}")

if __name__ == "__main__":
    unittest.main()
