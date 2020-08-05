import numpy as np
import cv2
from math import *
from matplotlib.pyplot import *

def debevecWeight(num):
    if num < 128:
        return num+1
    return 256-num

def g(images, times, index, i, j):
    b, g, r = images[index][i][j]
    lb, lg, lr = log(b), log(g), log(r)
    lt = log(times[index])
    return [lb-lt, lg-lt, lr-lt]


def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    # Need a high feature to save the color
    feature_num = 20000
    orb = cv2.ORB_create(feature_num)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches) * 0.6)]
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # wrap all pixels in one image to map it to the other
    height, width, _ = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    
    return im1Reg, h

def getResponseCurve(images, times):
    rows, cols, _ = images[0].shape
    res_split = np.zeros((3,256), np.float32)
    ret = np.zeros((256,1,3), np.float32)

    sample_x, sample_y = [0]*100, [0]*100
    col_num = int(sqrt(70*cols/rows))
    row_num = int(70/col_num)

    count = 0
    col = int((cols/col_num) / 2)
    for i in range(col_num):
        row = int((rows/row_num) / 2)
        for j in range(row_num):
            if (row>=0 and row<rows) and (col>=0 and col<cols):
                sample_x[count] = row
                sample_y[count] = col
                count += 1
            row += int(rows/row_num)
        col += int(cols/col_num)

    #solve response function for each channel
    for bgr in range(3):
        A = np.zeros((len(images)*count+257, 256+count), np.float32)
        B = np.zeros((len(images)*count+257, 1), np.float32)

        # data-fitting equation
        k = 0
        for i in range(count):
            for j in range(len(images)):
                # print(j, sample_x[i], sample_y[i], bgr)
                val = images[j][sample_x[i]][sample_y[i]][bgr]
                weight = debevecWeight(val)
                A[k][val] = weight
                A[k][256+i] = -weight
                B[k][0] = weight * log(times[j])
                k += 1

        #fix curve by setting mid to 0
        A[k][128] = 1
        k+=1

        #debevec smoothness equation
        for i in range(254):
            wi = debevecWeight(i+1)
            A[k][i] = 10*wi
            A[k][i+1] = -20 * wi
            A[k][i+2] = 10 * wi
            k += 1

        temp = np.zeros(256, np.float32)
        temp = cv2.solve(A, B, temp, cv2.DECOMP_SVD)[1]
        for i in range(256):
            res_split[bgr][i] = temp[i]

    for p in range(256):
        for bgr in range(3):
            ret[p][0][bgr] = exp(res_split[bgr][p])

    return ret


def readImagesAndTimes():
    # mountain shed
    # times = np.array([1/40,1/160,1/640,1/2500], dtype=np.float32)
    # mountain shed
    # filenames = ["input_mountain/input1.jpg", "input_mountain/input2.jpg", "input_mountain/input3.jpg", "input_mountain/input4.jpg"]


    # # List of exposure times
    times = np.array([ 30, 4, 1/2, 1/15], dtype=np.float32)
    # # List of image filenames
    filenames = ["../input/input1.png", "../input/input2.png", "../input/input3.png", "../input/input4.png"]
    
    images = []
    for filename in filenames:
        images.append(cv2.imread(filename))
  
    return images, times



def main():
    images, times = readImagesAndTimes()
    out_dir = "../output"
    # Align input images
    # alignMTB = cv2.createAlignMTB()
    # alignMTB.process(images, images)
    for i in range(1, len(images)):
        images[i-1],h = alignImages(images[i-1], images[i])
 

    # Obtain Camera Response Function (CRF)

    rc = getResponseCurve(images, times)
    for bgr in range(3):
        color = 'b-'
        if bgr == 1:
            color = 'g-'
        elif bgr == 2:
            color = 'r-'
        plot(range(256), rc[..., 0,bgr], color)
    xlabel('Pixel Value')
    ylabel('Exposure Value')
    legend(['B', 'G', 'R'])
    title('Camera Response Curve')
    savefig('CRC.png')


    # Merge images into an HDR linear image
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, rc)
    # Save HDR image.
    cv2.imwrite("hdrDebevec.hdr", hdrDebevec)

    tonemapDrago = cv2.createTonemapDrago(1, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    cv2.imwrite(out_dir+"/my_hdr.jpg", ldrDrago * 255)

     # Tonemap using Reinhard's method to obtain 24-bit color image
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    cv2.imwrite(out_dir+"/ldr-Reinhard.jpg", ldrReinhard * 255)

    # Tonemap using Mantiuk's method to obtain 24-bit color image
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    cv2.imwrite(out_dir+"/ldr-Mantiuk.jpg", ldrMantiuk * 255)

if __name__ == '__main__':
    main()