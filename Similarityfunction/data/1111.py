# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:59:22 2018

@author: zy
"""

'''
使用BRIEF特征描述符
'''
import cv2
import numpy as np


def brief_test():
    # 加载图片  灰色
    img1 = cv2.imread('C:\\Users\\kylenate\\Desktop\\test\\timg.jpg')
    img1 = cv2.resize(img1, dsize=(600, 400))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('C:\\Users\\kylenate\\Desktop\\test\\timg2.jpg')
    img2 = cv2.resize(img2, dsize=(600, 400))
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    image1 = gray1.copy()
    image2 = gray2.copy()

    image1 = cv2.medianBlur(image1, 5)
    image2 = cv2.medianBlur(image2, 5)

    '''
    1.使用SURF算法检测关键点
    '''
    # 创建一个SURF对象 阈值越高，能识别的特征就越少，因此可以采用试探法来得到最优检测。
    surf = cv2.xfeatures2d.SURF_create(3000)
    keypoints1 = surf.detect(image1)
    keypoints2 = surf.detect(image2)
    # 在图像上绘制关键点
    image1 = cv2.drawKeypoints(image=image1, keypoints=keypoints1, outImage=image1, color=(255, 0, 255),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2 = cv2.drawKeypoints(image=image2, keypoints=keypoints2, outImage=image2, color=(255, 0, 255),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # 显示图像
    cv2.imshow('sift_keypoints1', image1)
    cv2.imshow('sift_keypoints2', image2)
    cv2.waitKey(20)

    '''
    2.计算特征描述符
    '''
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(32)
    keypoints1, descriptors1 = brief.compute(image1, keypoints1)
    keypoints2, descriptors2 = brief.compute(image2, keypoints2)

    print('descriptors1:', len(descriptors1), 'descriptors2', len(descriptors2))
    print(descriptors1)

    '''
    3.匹配  汉明距离匹配特征点
    '''
    matcher = cv2.BFMatcher_create(cv2.HAMMING_NORM_TYPE)
    matchePoints = matcher.match(descriptors1, descriptors2)
    print(type(matchePoints), len(matchePoints), matchePoints[0])

    # 提取强匹配特征点
    minMatch = 1
    maxMatch = 0
    for i in range(len(matchePoints)):
        if minMatch > matchePoints[i].distance:
            minMatch = matchePoints[i].distance
        if maxMatch < matchePoints[i].distance:
            maxMatch = matchePoints[i].distance
    print('最佳匹配值是:', minMatch)
    print('最差匹配值是:', maxMatch)

    # 获取排雷在前边的几个最优匹配结果
    goodMatchePoints = []
    for i in range(len(matchePoints)):
        if matchePoints[i].distance < minMatch + (maxMatch - minMatch) / 3:
            goodMatchePoints.append(matchePoints[i])

    # 绘制最优匹配点
    outImg = None
    outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, goodMatchePoints, outImg, matchColor=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imshow('matche', outImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    brief_test()