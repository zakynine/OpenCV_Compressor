import time
import os
import csv

# Library for MSE SSIM Counting
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import argparse

# Library for PSNR
from math import log10, sqrt
import cv2
import numpy as np

def get_size_format(b, factor=1024, suffix="B"):
    """
    Scale bytes to its proper byte format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if b < factor:
            return f"{b:.2f}{unit}{suffix}"
        b /= factor
    return f"{b:.2f}Y{suffix}"

# Compression algorithm
import cv2
import numpy as np

def save(path, image, jpg_quality=None, png_compression=None):
  '''
  persist :image: object to disk. if path is given, load() first.
  jpg_quality: for jpeg only. 0 - 100 (higher means better). Default is 95.
  png_compression: For png only. 0 - 9 (higher means a smaller size and longer compression time).
                  Default is 3.
  '''

  if jpg_quality:
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
  #elif png_compression:
    #cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
  else:
    cv2.imwrite(path, image)

########## Start here are codes for MSE SSIM Measurement
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
	mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	mse_error /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE. The lower the error, the more "similar" the two images are.
	return mse_error

def compare(imageA, imageB):
	# Calculate the MSE and SSIM
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# Return the SSIM. The higher the value, the more "similar" the two images are.
	return s

########## Start here are codes for PSNR

def PSNR(image1, image2):
	mse = np.mean((image1 - image2) ** 2)
	if(mse == 0): # MSE is zero means no noise is present in the signal .
				# Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr
# declare list compression time
CT_record = []
CR_record = []
MSE_record = []
SSIM_record = []
PSNR_record = []


def main():

    # Initialize the current time as the start time
    start_time = time.time()
    
    # Call other functions related to the compression algorithm
        
    imgpath = "original_image.tiff"
    img = cv2.imread(imgpath)

    # save the image in JPEG format with variable quality, change for your desire quality

    outpath_jpeg = "compressed_image.jpg"
    quality = 95
    save(outpath_jpeg,img,jpg_quality=quality)

    cv2.waitKey(0)

    ############# Start here are codes for MSE SSIM Measurement in main()
	# Import images
    image1 = cv2.imread(imgpath)
    image2 = cv2.imread(outpath_jpeg)

	# Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

	# Check for same size and ratio and report accordingly
    ho, wo, _ = image1.shape
    hc, wc, _ = image2.shape
    ratio_orig = ho/wo
    ratio_comp = hc/wc
    dim = (wc, hc)

    if round(ratio_orig, 2) != round(ratio_comp, 2):
        print("\nImages not of the same dimension. Check input.")
        exit()

	# Resize first image if the second image is smaller
    elif ho > hc and wo > wc:
        print("\nResizing original image for analysis...")
        gray1 = cv2.resize(gray1, dim)

    elif ho < hc and wo < wc:
        print("\nCompressed image has a larger dimension than the original. Check input.")
        exit()

    if round(ratio_orig, 2) == round(ratio_comp, 2):

        mse_value = mse(gray1, gray2)
        ssim_value = compare(gray1, gray2)
        print("[*] Quality:", quality)
        print("[+] MSE:", mse_value)
        print("[+] SSIM:", ssim_value)
       

    # get the original image size in bytes
    image_size = os.path.getsize(imgpath)
    #print("[*] Size before compression:", get_size_format(image_size))

    # get the new image size in bytes
    new_image_size = os.path.getsize(outpath_jpeg)
    #print("[+] Size after compression:", get_size_format(new_image_size))

    # Print Compression Ratio
    compression_ratio = (new_image_size) / (image_size)
    print("[+] Compression Ration :", compression_ratio)

    # Get PSNR Value
    psnr_value = PSNR(image1, image2)
    print(f"[+]PSNR : {psnr_value} dB")
   
    # Call compression_time function to get the compression time
    
    processing_time = time.time() - start_time
    print("[*] Compression Time :", processing_time)
    print("==========================================")
    

    MSE_record.append(mse_value)
    SSIM_record.append(ssim_value)
    PSNR_record.append(psnr_value)
    CR_record.append(compression_ratio)
    CT_record.append(processing_time)

    print('==== MSE LIST =======', MSE_record)
    print('==== SSIM LIST =======', SSIM_record)
    print('==== PSNR LIST =======', PSNR_record)
    print('==== CR LIST =======', CR_record)
    print('==== CT LIST =======', CT_record)

    print("==========================================", i)

    # Give delay 
    time.sleep(3)

for i in range(1,34) :
    if i == 34: 
        break
    if __name__ == "__main__":
        main()