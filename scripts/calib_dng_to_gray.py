import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
import imageio
import os
import glob


def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('inputs', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args


def black_subtraction(img, black, white):

    proc_im = img.astype(np.float32) - black
    proc_im = np.clip(proc_im, 0, white).astype(np.uint16)

    return proc_im

def color_scaling(img, wb):

    R = img[1::2, 0::2].astype(np.float32) * wb[0]
    B = img[0::2, 1::2].astype(np.float32) * wb[2]
    G1 = img[0::2, 0::2].astype(np.float32) * wb[1]
    G2 = img[1::2, 1::2].astype(np.float32) * wb[1]
    
    proc_im = img.copy()
    proc_im[1::2, 0::2] = np.clip(R, 0, 65535).astype(np.uint16)
    proc_im[0::2, 1::2] = np.clip(B, 0, 65535).astype(np.uint16)
    proc_im[0::2, 0::2] = np.clip(G1, 0, 65535).astype(np.uint16)
    proc_im[1::2, 1::2] = np.clip(G2, 0, 65535).astype(np.uint16)

    return proc_im

def apply_CCM(img, CCM):

    img_float = img.astype(np.float32)
    ccm = CCM[:, :3]

    # 1. Reshape image to (N, 3) where N is total number of pixels
    pixels = img_float.reshape(-1, 3)

    # 2. Multiply pixels by CCM (using @ for matrix multiplication)
    # Use ccm.T if your CCM is defined to multiply from the right: pixels @ ccm.T
    corrected_pixels = pixels @ ccm.T 

    # 3. Reshape back and clip values to [0, 1] range
    corrected_img = corrected_pixels.reshape(img.shape)
    corrected_img = np.clip(corrected_img, 0, 65535).astype(np.uint16)

    return corrected_img

def apply_gamma(img, gamma=2.2, linear_slope=4.5):
    # 1. Load image and convert to YCbCr (standard YUV for OpenCV)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, u, v = cv2.split(img_yuv)

    # 2. Normalize Luma channel to [0, 1]
    y_norm = y / 255.0

    # 3. Piecewise Gamma Correction (Linear ends + Gamma center)
    # Similar to sRGB/BT.709 transfer function
    y_corr = np.where(y_norm <= 0.018, 
                      y_norm * linear_slope, 
                      (1.0 + 0.055) * np.power(y_norm, 1/gamma) - 0.055)

    # 4. Scale back to [0, 255] and replace Y channel
    y_new = np.clip(y_corr * 255.0, 0, 255).astype(np.uint8)
    
    # 5. Merge and convert back to BGR
    img_yuv_new = cv2.merge((y_new, u, v))
    result = cv2.cvtColor(img_yuv_new, cv2.COLOR_YCrCb2BGR)
    return result


def main():
    #code
    files = glob.glob(args.inputs)

    for file in files:
        with rawpy.imread(file) as raw:
            # Post-process to get an RGB array
            print(f'Raw type: {raw.raw_type}')
            print(f'Number of colors: {raw.num_colors}')
            print(f'Color description: {raw.color_desc}')
            print(f'Color Matrix:\n {raw.color_matrix}')
            print(f'Black levels: {raw.black_level_per_channel}')
            print(f'White level: {raw.white_level}')
            print(f'Camera white balance: {raw.camera_whitebalance}')
    
            black = raw.black_level_per_channel
            white = raw.white_level
            ccm = raw.color_matrix
            wb = raw.camera_whitebalance
            #raw_im = raw.raw_image_visible.astype(np.uint16)
    
            #Demosaicing
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, auto_bright_thr=0.035, use_camera_wb=True, output_bps=16, 
                                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.sRGB)
            
    
        rgb = (rgb.astype(np.float32) / 2**16)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ceiling = 35.0
        ldr_8bit = np.clip((gray.astype(np.float32)*255)*255/ceiling, 0, 255).astype('uint8')
        #ldr_8bit = cv2.cvtColor(ldr_8bit, cv2.COLOR_BGR2GRAY)
        print(ldr_8bit.shape)
    
        # Plotting results
        #plt.figure(1)
        #plt.imshow(ldr_8bit, cmap='gray')
        #plt.show()
    
        print("\nSaving Image Output...\n")
        input_filename = os.path.basename(file)
        input_filename = os.path.splitext(input_filename)[0]
        output_filenameL = f"out_gL_{input_filename}.png"
        output_filenameR = f"out_gR_{input_filename}.png"
        output_filenameLR = f"out_gLR_{input_filename}.png"
        #print("Output Filename: ", output_filename)
        #imageio.imsave(output_filenameR, ldr_8bit[:, int(ldr_8bit.shape[1]/2):], quality=100)
        #imageio.imsave(output_filenameL, ldr_8bit[:, :int(ldr_8bit.shape[1]/2)], quality=100)
        imageio.imwrite(output_filenameL, ldr_8bit[:, :int(ldr_8bit.shape[1]/2)])
        imageio.imwrite(output_filenameR, ldr_8bit[:, int(ldr_8bit.shape[1]/2):])
        #imageio.imwrite(output_filenameLR, ldr_8bit)


if __name__=="__main__":
    args = parse()
    main()