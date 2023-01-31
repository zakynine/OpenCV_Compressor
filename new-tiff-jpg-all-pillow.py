import os
from PIL import Image
import time

# Library for MSE SSIM Counting
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

# Library for PSNR
from math import log10, sqrt
import cv2
import numpy as np



# Compression algorithm
# Get size file
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
 

def compress_img(image_name, new_size_ratio=0.9, quality=90, width=None, height=None, to_jpg=True):
    # load the image to memory
    img = Image.open(image_name)

    if new_size_ratio < 1.0:
        # if resizing ratio is below 1.0, then multiply width & height with this ratio to reduce image size
        img = img.resize((int(img.size[0] * new_size_ratio), int(img.size[1] * new_size_ratio)), Image.ANTIALIAS)
        # print new image shape
        print("[+] New Image shape:", img.size)
    elif width and height:
        # if width and height are set, resize with them instead
        img = img.resize((width, height), Image.ANTIALIAS)
        # print new image shape
        print("[+] New Image shape:", img.size)
    # split the filename and extension
    filename, ext = os.path.splitext(image_name)
    # make new filename appending _compressed to the original file name
    if to_jpg:
        # change the extension to JPEG
        new_filename = f"{filename}_compressed.jpg"
    else:
        # retain the same extension of the original image
        new_filename = f"{filename}_compressed{ext}"
    try:
        # save the image with the corresponding quality and optimize set to True
        img.save(new_filename, quality=quality, optimize=True)
    except OSError:
        # convert the image to RGB mode first
        img = img.convert("RGB")
        # save the image with the corresponding quality and optimize set to True
        img.save(new_filename, quality=quality, optimize=True) 


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

################################## ==== MAIN ===== #######################

def main():
    start_time = time.time()

    # Call other functions related to the compression algorithm
    import argparse
    parser = argparse.ArgumentParser(description="Simple Python script for compressing and resizing images")
    parser.add_argument("image", help="Target image to compress and/or resize")
    parser.add_argument("-j", "--to-jpg", action="store_true", help="Whether to convert the image to the JPEG format")
    parser.add_argument("-q", "--quality", type=int, help="Quality ranging from a minimum of 0 (worst) to a maximum of 95 (best). Default is 90", default=90)
    parser.add_argument("-r", "--resize-ratio", type=float, help="Resizing ratio from 0 to 1, setting to 0.5 will multiply width & height of the image by 0.5. Default is 1.0", default=1.0)
    parser.add_argument("-w", "--width", type=int, help="The new width image, make sure to set it with the `height` parameter")
    parser.add_argument("-hh", "--height", type=int, help="The new height for the image, make sure to set it with the `width` parameter")
    args = parser.parse_args()
    # compress the image
    compress_img(args.image, args.resize_ratio, args.quality, args.width, args.height, args.to_jpg)

############# Start here are codes for MSE SSIM Measurement in main()
	# Import images
    img_ori_path = "original_image.tiff"
    img_comp_path = "original_image_compressed.jpg"

    image1 = cv2.imread(img_ori_path)
    image2 = cv2.imread(img_comp_path)

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
        print("MSE:", mse_value)
        print("SSIM:", ssim_value)
    

    ##################### Start here are codes for PSNR 
    # Get PSNR Value
    psnr_value = PSNR(image1, image2)
    print(f"[+]PSNR : {psnr_value} dB")

    print("="*50)
    
    
    # get the original image size in bytes
    image_size = os.path.getsize(img_ori_path)
    new_image_size = os.path.getsize(img_comp_path)
    compression_ratio = (new_image_size) / (image_size)
    print("[+] Compression Ration :", compression_ratio)

    # Call compression_time function to get the compression time
    processing_time = time.time() - start_time
    print("[*] Time Compress:", processing_time)
    # print the passed arguments
    #print("[*] Image:", args.image)
    print("[*] To JPEG:", args.to_jpg)
    print("[*] Quality:", args.quality)
    #print("[*] Resizing ratio:", args.resize_ratio)

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
    print("="*50, i)

    # Give delay 
    time.sleep(3)

for i in range(1,34) :
    if i == 34: 
        break
    if __name__ == "__main__":
        main()


# ======================================== USE THIS COMMAND LINE FOR TESTING

# ... foldername> python new-tiff-jpg-all-pillow.py -j -q 90