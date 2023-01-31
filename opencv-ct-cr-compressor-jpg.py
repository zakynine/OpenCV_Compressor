import time
import os
# Initialize the current time as the start time
start_time = time.time()

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

# Get the compression time
def compression_time():
    # Return the compression time
    return (time.time() - start_time)

def main():
    # Call other functions related to the compression algorithm
        
    imgpath = "original_image.tiff"
    img = cv2.imread(imgpath)

    #display the image
    #cv2.imshow('original_image', img)

    # save the image in JPEG format with variable quality, change for your desire quality
    outpath_jpeg = "compressed_image.jpg"
    quality = 100
    save(outpath_jpeg,img,jpg_quality=quality)

    # outpath_png = "compressed_image.png"

    # save the image in PNG format with 4 Compression
    # save(outpath_png, img,png_compression=4)

    cv2.waitKey(0)
    #destroy a certain window
    #cv2.destroyWindow('original_image')
    
    print("[*] Quality:", quality)
    # get the original image size in bytes
    image_size = os.path.getsize(imgpath)
    print("[*] Size before compression:", get_size_format(image_size))
    # get the new image size in bytes
    new_image_size = os.path.getsize(outpath_jpeg)
    print("[+] Size after compression:", get_size_format(new_image_size))
    # Print Compression Ratio
    compression_ratio = (new_image_size) / (image_size)
    print("[+] Compression Ration:", compression_ratio)

    # Call compression_time function to get the compression time
    processing_time = compression_time()
    print("[*] Time Compress:", processing_time)

if __name__ == "__main__":
    main()