from PIL import Image
from numpy import average, linalg, dot
import numpy as np
import imagehash
from skimage.metrics import structural_similarity

__author__ = 'jianxing.wei@wuage.com '

from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

class ImageDiff():
    def __init__(self,imageA_dir, imageB_dir):
        self.srcA = imageA_dir
        self.srcB = imageB_dir
        self.imageA = cv2.imread(imageA_dir)
        self.imageB = cv2.imread(imageB_dir)
        # 尺寸冗余调整
        h_a, w_a = self.imageA.shape[:2]
        h_b, w_b = self.imageB.shape[:2]

        # h = int(max(h_a, h_b))
        # w = int(max(w_a, w_b))

        h = int(max(h_a, h_b))
        w = int(max(w_a, w_b))

        # h = max(h_a, h_b)
        # w = max(w_a, w_b)
        self.resizeA = cv2.resize(self.imageA, dsize=(w, h))
        self.resizeB = cv2.resize(self.imageB, dsize=(w, h))
        cv2.imwrite("resizeA.jpg", self.resizeA)
        cv2.imwrite("resizeB.jpg", self.resizeB)
        # 转换成shape为2的灰阶图片
        self.grayA = cv2.cvtColor(self.resizeA, cv2.COLOR_BGR2GRAY)
        self.grayB = cv2.cvtColor(self.resizeB, cv2.COLOR_BGR2GRAY)

        cv2.imwrite("grayA.jpg", self.grayA)
        cv2.imwrite("grayB.jpg", self.grayB)

    # def setImage(self,imageA_dir, imageB_dir):
    #     self.imageA = cv2.imread(imageA_dir)
    #     self.imageB = cv2.imread(imageB_dir)
    #     # 尺寸冗余调整
    #     h_a, w_a = self.imageA.shape[:2]
    #     h_b, w_b = self.imageB.shape[:2]
    #
    #     h = min(h_a, h_b)
    #     w = min(w_a, w_b)
    #     # self.resizeA = cv2.resize(self.imageA, dsize=(w, h), fx=h_a / h, fy=w_a / w)
    #     # self.resizeB = cv2.resize(self.imageB, dsize=(w, h), fx=h_b / h, fy=w_b / w)
    #
    #     self.resizeA = cv2.resize(self.imageA, dsize=(w, h))
    #     self.resizeB = cv2.resize(self.imageB, dsize=(w, h))
    #     cv2.imwrite("resizeA.jpg", self.resizeA)
    #     cv2.imwrite("resizeB.jpg", self.resizeB)
    #     # 转换成shape为2的灰阶图片
    #     self.grayA = cv2.cvtColor(self.resizeA, cv2.COLOR_BGR2GRAY)
    #     self.grayB = cv2.cvtColor(self.resizeB, cv2.COLOR_BGR2GRAY)
    def diffImages(self):
        """
        he score  represents the structural similarity index between the two input images. This value can fall into the range [-1, 1]
        with a value of one being a “perfect match”.
        The diff  image contains the actual image differences between the two input images that we wish to visualize.
        The difference image is currently represented as a floating point data type in the range [0, 1]
        so we first convert the array to 8-bit unsigned integers in the range [0, 255]  before we can further process it using OpenCV.
        :return:
        """
        (score, diff) = compare_ssim(self.grayA, self.grayB, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        # cv2.THRESH_BINARY, 48, 8)
        # cv2.CHAIN_APPROX_TC89_L1
        #cv2.RETR_EXTERNAL
        #返回轮廓集合
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(self.resizeA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.rectangle(self.resizeB, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.rectangle(self.grayA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(self.grayB, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # show the output images
        # cv2.imshow("Original", self.imageA)
        # cv2.imshow("Modified", self.imageB)
        cv2.imwrite("results.jpg",self.grayB)
        # cv2.imshow("Diff", diff)
        # cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)

    def diffImageswithdir(self, imageA_dir, imageB_dir):
        self.imageA = cv2.imread(imageA_dir)
        self.imageB = cv2.imread(imageB_dir)
        # 转换成shape为2的灰阶图片
        self.grayA = cv2.cvtColor(self.imageA, cv2.COLOR_BGR2GRAY)
        self.grayB = cv2.cvtColor(self.imageB, cv2.COLOR_BGR2GRAY)

    # 对图片进行统一化处理
    def get_thum(self, image, size=(128, 128), greyscale=False):
        # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
        image = image.resize(size, Image.ANTIALIAS)
        # image = cv2.resize(image, dsize=size)

        if greyscale:
            # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
            image = image.convert('L')
        return image
    # 计算图片的余弦距离
    def image_similarity_vectors_via_numpy(self, image1 , image2):
        image1 = self.get_thum(Image.open(image1))
        image2 = self.get_thum(Image.open(image2))
        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.getdata():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            # linalg=linear（线性）+algebra（代数），norm则表示范数
            # 求图片的范数？？
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        # dot返回的是点积，对二维数组（矩阵）进行计算
        res = dot(a / a_norm, b / b_norm)
        return res

    # 计算两个哈希值之间的差异 汉明距离计算
    def campHash(self, hash1, hash2):
        n = 0
        # hash长度不同返回-1,此时不能比较
        if len(hash1) != len(hash2):
            return -1
        # 如果hash长度相同遍历长度
        for i in range(len(hash1)):
            if hash1[i] != hash2[i]:
                n = n + 1
        return n

    # 差异值哈希算法
    def dhash(self , image):
        # 将图片转化为8*8
        image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
        # 将图片转化为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dhash_str = ''
        for i in range(8):
            for j in range(8):
                if gray[i, j] > gray[i, j + 1]:
                    dhash_str = dhash_str + '1'
                else:
                    dhash_str = dhash_str + '0'
        result = ''
        for i in range(0, 64, 4):
            result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
        # print("dhash值",result)
        return result

    # phash
    def phash(self,path):
        # 加载并调整图片为32*32的灰度图片
        img = cv2.imread(path)
        img1 = cv2.resize(img, (32, 32), cv2.COLOR_RGB2GRAY)
        # 创建二维列表
        h, w = img.shape[:2]
        vis0 = np.zeros((h, w), np.float32)
        vis0[:h, :w] = img1
        # DCT二维变换
        # 离散余弦变换，得到dct系数矩阵
        img_dct = cv2.dct(cv2.dct(vis0))
        img_dct.resize(8, 8)
        # 把list变成一维list
        img_list = np.array().flatten(img_dct.tolist())
        # 计算均值
        img_mean = cv2.mean(img_list)
        avg_list = ['0' if i < img_mean else '1' for i in img_list]
        return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 64, 4)])

    def diff_phash(self, image1, image2,size = 8):
        # i1 = imagehash.phash(Image.open(image1), size)
        # i2 = imagehash.phash(Image.open(image2), size)
        # PIL.Image.fromarray(cv2.imread('a.jpg'))
        i1 = imagehash.phash(Image.fromarray(cv2.imread(image1)), size)
        i2 = imagehash.phash(Image.fromarray(cv2.imread(image2)), size)
        print(i1)
        print(i2)
        return self.campHash(str(i1), str(i2))

    def erode_image(self, img_path):
        origin_img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        # OpenCV定义的结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 腐蚀图像
        eroded = cv2.erode(gray_img, kernel)
        return eroded
        # 显示腐蚀后的图像
        # cv2.imshow('Origin', origin_img)
        # cv2.imshow('Erode', eroded)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def dilate_image(self, img_path):
        origin_img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
        # OpenCV定义的结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 膨胀图像
        dilated = cv2.dilate(gray_img, kernel)
        # 显示腐蚀后的图像
        cv2.imshow('Dilate', dilated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getDilate_image(self, gray_img):
        # OpenCV定义的结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # 膨胀图像
        dilated = cv2.dilate(gray_img, kernel)
        return dilated
    def getErode_image(self, gray_img):
        # OpenCV定义的结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # 腐蚀图像
        eroded = cv2.erode(gray_img, kernel)
        return eroded

    def iter_erodeDilateImage(self,image, times):

        mix = image
        for i in range(times):
            dilate = self.getDilate_image(mix)
            mix =  self.getErode_image(dilate)
        return mix
    def diffImagesWithErodeAndDilate(self):
        a = self.getDilate_image(self.grayA)
        b = self.getDilate_image(self.grayB)

        mixA = self.iter_erodeDilateImage(a,2)
        mixB = self.iter_erodeDilateImage(b,2)

        (score, diff) = compare_ssim(mixA, mixB, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        # cv2.THRESH_BINARY, 48, 8)
        # cv2.CHAIN_APPROX_TC89_L1
        #cv2.RETR_EXTERNAL
        #返回轮廓集合
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            # cv2.rectangle(self.resizeA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.rectangle(self.resizeB, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.rectangle(mixA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.rectangle(mixB, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # show the output images
        # cv2.imshow("Original", self.imageA)
        # cv2.imshow("Modified", self.imageB)
        cv2.imwrite("results.jpg",mixA)
        # cv2.imshow("Diff", diff)
        # cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)
    def structDiff(self):
        # Compute SSIM between two images
        a = self.getDilate_image(self.grayA)
        b = self.getDilate_image(self.grayB)

        mixA = self.getErode_image(a)
        mixB = self.getErode_image(b)
        (score, diff) = structural_similarity(mixA, mixB, full=True)
        print("Image similarity", score)

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1]
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")

        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        mask = np.zeros(mixA.shape, dtype='uint8')
        filled_after = mixB.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(mixA, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(mixB, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

        cv2.imshow('before', mixA)
        cv2.imshow('after', mixB)
        cv2.imshow('diff', diff)
        cv2.imshow('mask', mask)
        cv2.imshow('filled after', filled_after)
        cv2.waitKey(0)