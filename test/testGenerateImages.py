import unittest

from utils.diffImages import ImageDiff

__author__ = 'jianxing.wei '
from selenium import webdriver

class TestDiff(unittest.TestCase):
    def setUp(self):
        print("testing diff image.")
        self.diff = ImageDiff("test_full_chrome.png", "test_full_firefox.png")
    def testSnapshot(self):

        # self.diff.diffImages()
        # sim_res = self.diff.diff_phash("test_full_chrome.png", "test_full_firefox.png",24)
        # print(sim_res)
        # sim_res = self.diff.image_similarity_vectors_via_numpy("resizeA.jpg", "resizeB.jpg")
        # sim_res = self.diff.diff_phash("resizeA.jpg", "resizeB.jpg",24)
        self.diff.diffImagesWithErodeAndDilate()

        # print(sim_res)

if __name__ == '__main__':
    unittest.main()