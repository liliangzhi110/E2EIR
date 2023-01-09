import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import random

import matplotlib.pyplot as plt


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF
    Run this routine for the given parameters patch_width = 9 and n = 256
    INPUTS
    patch_width - the width of the image patch (usually 9)
    nbits      - the number of tests n in the BRIEF descriptor
    OUTPUTS
    compareX and compareY - LINEAR indices into the patch_width x patch_width image
                            patch and are each (nbits,) vectors.
    '''
    #############################
    # TO DO ...
    # Generate testpattern here

    # RANGE OF COMPARE X AND Y SHOULD BE BETWEEN 0 AND 81

    # IT IS GIVEN IN THE PAPER THAT sigma*sigma = 1/25 (S*S), WHERE S IS 9. S*S IS SAMPLE SPACE
    # SO, SIGMA EQUALS 9/5
    sigma, k = 1.8, 9  # generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = sigma
    probs = [np.exp(-z * z / (2 * sigma * sigma)) / np.sqrt(2 * np.pi * sigma * sigma) for z in range(-4, 4 + 1)]
    kernel = np.outer(probs, probs)
    kernel = np.multiply(kernel, 256)
    kernel = kernel.round()
    # print(kernel)
    # print(kernel.sum())

    half = (int)(patch_width / 2)

    kernel = kernel.astype(int)

    kernel[half, half] -= 1
    kernel[half, half - 1] -= 1
    kernel[half, half + 1] -= 1
    kernel[half - 1, half] -= 1
    kernel[half + 1, half] -= 1

    # print(kernel)
    # print(kernel.sum())

    # plt.imshow(kernel)
    # plt.colorbar()
    # plt.show()

    compareX, compareY = [], []

    counter = 0
    for i in range(0, patch_width):
        for j in range(0, patch_width):
            for k in range(0, kernel[i, j]):
                compareX.append(counter)
                compareY.append(counter)
            counter += 1

    # print(compareX)
    # print(compareY)

    random.shuffle(compareX)
    random.shuffle(compareY)

    print('compareX: ', compareX)
    print('compareY: ', compareY)

    test_pattern_file = '../results/testPattern.npy'
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])
    print('test_pattern_file saved')

    return compareX, compareY


# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
    print('loaded compare X and Y from results ')
    # print('loaded compare X: ', compareX)
    # print('loaded compare Y: ', compareY)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])
    print('test_pattern_file saved2')

    compareX, compareY = np.load(test_pattern_file)


def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
                 compareX, compareY):
    '''
    Compute Brief feature
     INPUT
     locsDoG - locsDoG are the keypoint locations returned by the DoG
               detector.
     levels  - Gaussian scale levels that were given in Section1.
     compareX and compareY - linear indices into the
                             (patch_width x patch_width) image patch and are
                             each (nbits,) vectors.


     OUTPUT
     locs - an m x 3 vector, where the first two columns are the image
    		 coordinates of keypoints and the third column is the pyramid
            level of the keypoints.
     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
            of valid descriptors in the image and will vary.
    '''
    ##############################
    # TO DO ...
    # compute locs, desc here
    locs = []
    desc = []

    # print('locsDoG shape: ', locsDoG.shape)
    H = im.shape[0]
    W = im.shape[1]

    for key_point in locsDoG:
        column = key_point[0]
        row = key_point[1]
        depth = key_point[2]
        k_half = int(k / 2)
        if row >= k_half and column >= k_half and row <= (H - 1) - k_half and column <= (W - 1) - k_half:
            # WE TAKE DEPTH +1 BECAUSE WE HAVE COMPARE INTENSITY VALUES IN A SMOOTHED IMAGE (6 LEVELS ,  SMOOTHED LEVELS)
            patch = gaussian_pyramid[row - k_half:row + k_half + 1, column - k_half:column + k_half + 1, depth + 1]
            patch = np.asarray(patch)
            compare_vector = []
            patch_flat = patch.flatten()
            # print(patch_flat.shape)
            for i in range(0, len(compareX)):
                compare_vector.append(int(patch_flat[compareX[i]] < patch_flat[compareY[i]]))
            locs.append(key_point)
            desc.append(compare_vector)

    locs = np.asarray(locs)
    desc = np.asarray(desc)
    print('locs shape: ', locs.shape)
    print('desc shape: ', desc.shape)

    return locs, desc


def briefLite(im):
    '''
    INPUTS
    im - gray image with values between 0 and 1
    OUTPUTS
    locs - an m x 3 vector, where the first two columns are the image coordinates
            of keypoints and the third column is the pyramid level of the keypoints
    desc - an m x n bits matrix of stacked BRIEF descriptors.
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    ###################
    # TO DO ...

    # GRAYSCALE AND NORMALIZATION HANDELED IN CODE OF GAUSSIAN PYRAMID
    locsDoG, gaussian_pyramid = DoGdetector(im)
    print('locsDoG shape:  ', locsDoG.shape)

    compareX, compareY = np.load(test_pattern_file)

    locs, desc = computeBrief(im, gaussian_pyramid, locsDoG, k=9, levels=gaussian_pyramid.shape[2],
                              compareX=compareX, compareY=compareY)
    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    outputs : matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:, 0:2]
    d2 = d12.max(1)
    r = d1 / (d2 + 1e-10)
    is_discr = r < ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1, ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1] + im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i, 0], 0:2]
        pt2 = locs2[matches[i, 1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x, y, 'r', linewidth=0.8)
        plt.plot(x, y, 'g.')
    plt.show()


if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()

    # # -------------------------  TEST BRIEF_LIGHT  ---------------------------------
    # im = cv2.imread('../data/model_chickenbroth.jpg')
    # locs, desc = briefLite(im)
    # fig = plt.figure()
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    # plt.plot(locs[:,0], locs[:,1], 'r.')
    # plt.draw()
    # plt.waitforbuttonpress(0)
    # plt.close(fig)

    # # -------------------------  TEST MATCHING OF SAME IMAGES -----------------------
    # im = cv2.imread('../data/model_chickenbroth.jpg')
    # locs, desc = briefLite(im)
    # matches = briefMatch(desc, desc)
    # plotMatches(im,im,matches,locs,locs)

    # ---------------------- TEST MATCHING OF DIFFERENT IMAGES  ---------------------------------------------
    print('-------------------------   IMAGE 1  -----------------------------')
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    # im1 = cv2.imread('../data/incline_L.png')
    # im1 = cv2.imread('../data/pf_scan_scaled.jpg')
    locs1, desc1 = briefLite(im1)

    print('-------------------------   IMAGE 2  -----------------------------')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    # im2 = cv2.imread('../data/incline_R.png')
    # im2 = cv2.imread('../data/pf_desk.jpg')
    # im2 = cv2.imread('../data/pf_floor.jpg')
    # im2 = cv2.imread('../data/pf_pile.jpg')
    # im2 = cv2.imread('../data/pf_stand.jpg')
    # im2 = cv2.imread('../data/pf_floor_rot.jpg')
    locs2, desc2 = briefLite(im2)

    matches = briefMatch(desc1, desc2)
    print('matches shape: ', matches.shape)
    plotMatches(im1, im2, matches, locs1, locs2)