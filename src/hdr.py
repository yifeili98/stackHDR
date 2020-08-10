import numpy as np
import cv2
from myAlign import *
from myToneMap import *
from math import *
from matplotlib.pyplot import *
import time

def debevecWeight(num):
    if num < 128:
        return num+1
    return 256-num

def debevecWeightArr(arr):
    ret = []
    for num in arr:
        ret.append(debevecWeight(num))
    return np.array(ret)

def getResponseCurve(images, times):
    rows, cols, _ = images[0].shape
    res_split = np.zeros((3,256), np.float32)
    ret = np.zeros((256,1,3), np.float32)

    sample_x, sample_y = [0]*100, [0]*100
    col_num = int(sqrt(20*cols/rows))
    row_num = int(20/col_num)

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
            ret[p][0][bgr] = res_split[bgr][p]

    return ret

def radianceMapReconstruction(images, times, responseCurve):
    rows, cols, bgrs = images[0].shape
    ret = np.zeros(images[0].shape, np.float32)
    size = (rows, cols)

    # construct weight table for faster process
    weights = np.zeros((256, 1), np.float32)
    for i in range(256):
        weights[i] = i+1 if i<128 else 256-i

    # modify parameters for radiance mapping
    ln_time = np.log(times)
    result = np.zeros(images[0].shape, np.float32)
    result_split = cv2.split(result)
    weight_sum = np.zeros(size, np.float32)

    for j in range(len(images)):
        split_img = cv2.split(images[j])
        w = np.zeros(size, np.float32)

        for bgr in range(bgrs):
            split_img[bgr] = cv2.LUT(split_img[bgr], weights)
            w += split_img[bgr]/3

        response_img = cv2.LUT(images[j], responseCurve)
        split_img = cv2.split(response_img)
        for bgr in range(bgrs):
            m = np.multiply(w, split_img[bgr]-ln_time[j])
            result_split[bgr] += m
        weight_sum += w
    weight_sum = np.true_divide(1., weight_sum)

    for bgr in range(bgrs):
        result_split[bgr] = np.multiply(result_split[bgr], weight_sum)

    result = np.exp(cv2.merge(result_split))
    return result

def convert(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)

def readImagesAndTimes(filename):
    images = []
    times = [] # np.array([], dtype=np.float32)
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            content = line.split(" ")
            images.append(cv2.imread(content[0]))
            #times = np.append(times, convert(content[1]))
            times.append(convert(content[1]))

    if images == [] and times == []:
        print("Error when retrieving image file info!")
    return images, times



def main():
    assert filename != None
    assert out_dir != None
    print("Starting image alignment...")
    images, times = readImagesAndTimes(filename)
    # images = imageAlign(images)
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
    print("Image alignment successful, starting CRC reconstruction...")

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
    ylabel('Log Irradiance')
    legend(['B', 'G', 'R'])
    title('Camera Response Curve')
    savefig('CRC.png')
    print("Camera Response Curve is ready to view at cwd/CRC.png")
    
    
    print("Starting radiance mapping...")
    hdr = radianceMapReconstruction(images, times, rc)
    print(hdr)
    cv2.imwrite("radiance.hdr", hdr)
    print("Radiance mapping complete!")
    print("=========END OF IMAGE FUSION=========\n\n")

    print("Processing my tone map...")
    mytone = initiateDragoToneMapping(hdr)
    cv2.imwrite(out_dir+"/my_hdr.jpg", mytone * 765)

    print("Processing Drago's tone map...")
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdr)
    ldrDrago = 3 * ldrDrago
    cv2.imwrite(out_dir+"/ldr-Drago.jpg", ldrDrago * 255)

    print("Processing Reinhard's tone map...")
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
    ldrReinhard = tonemapReinhard.process(hdr)
    cv2.imwrite(out_dir+"/ldr-Reinhard.jpg", ldrReinhard * 255)

    print("Processing Mantiuk's tone map...")
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdr)
    ldrMantiuk = 3 * ldrMantiuk
    cv2.imwrite(out_dir+"/ldr-Mantiuk.jpg", ldrMantiuk * 255)

    # construct radiance map visualization
    print("Constructing radiance heat map...")
    result = cv2.imread("radiance.hdr")
    rad_map = np.uint8(np.log(result)) + 1
    norm = cv2.normalize(rad_map, None, 0, 190, cv2.NORM_MINMAX, cv2.CV_8UC1)
    fc = cv2.applyColorMap(norm, cv2.COLORMAP_JET) 
    cv2.imwrite("recover.png", fc)

if __name__ == '__main__':
    filename = str(sys.argv[1])
    out_dir = str(sys.argv[2])
    main()