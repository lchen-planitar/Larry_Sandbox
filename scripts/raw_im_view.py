import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
import imageio
import os


def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args


def LCA_correction(image, r_scale, b_scale):
    r_image = image[1::2, 0::2]
    b_image = image[0::2, 1::2]

    (h, w) = r_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, r_scale)
    r_image_scaled = cv2.warpAffine(r_image, M, (w, h))

    (h, w) = b_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, b_scale)
    b_image_scaled = cv2.warpAffine(b_image, M, (w, h))

    out_image = image.copy()
    out_image[1::2, 0::2] = r_image_scaled
    out_image[0::2, 1::2] = b_image_scaled

    return out_image

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

def RG_delta(img):

    R = img[1::2, 0::2].astype(np.float32)
    B = img[0::2, 1::2].astype(np.float32)
    G1 = img[0::2, 0::2].astype(np.float32)
    G2 = img[1::2, 1::2].astype(np.float32)
    
    delta = R-(G1+G2)/2
    delta = np.clip(delta, 0, 65535).astype(np.uint16)

    return delta

def BG_delta(img):

    R = img[1::2, 0::2].astype(np.float32)
    B = img[0::2, 1::2].astype(np.float32)
    G1 = img[0::2, 0::2].astype(np.float32)
    G2 = img[1::2, 1::2].astype(np.float32)
    
    delta = B-(G1+G2)/2
    delta = np.clip(delta, 0, 65535).astype(np.uint16)

    return delta

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

def sharpen_raw(img, factor):
    # Sharpening kernel
    #factor = 0.1
    kernel = np.array([[0, -factor, 0], 
                       [-factor, 1+4*factor, -factor], 
                       [0, -factor, 0]])
    # Apply the kernel to the image
    sharpened = cv2.filter2D(img, -1, kernel)
    sharpened = np.clip(sharpened, 0, 65535).astype(np.uint16)
    return sharpened

def sharpen(img, factor):
    # Sharpening kernel
    #factor = 0.1
    kernel = np.array([[0, -factor, 0], 
                       [-factor, 1+4*factor, -factor], 
                       [0, -factor, 0]])
    # Apply the kernel to the image
    sharpened = cv2.filter2D(img, -1, kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def axial_CA_correction(img, factorR, factorB):
    R = img[1::2, 0::2].astype(np.float32)
    B = img[0::2, 1::2].astype(np.float32)
    G1 = img[0::2, 0::2].astype(np.float32)
    G2 = img[1::2, 1::2].astype(np.float32)

    proc_im = img.copy()
    proc_im[1::2, 0::2] = sharpen_raw(R, factorR)
    proc_im[0::2, 1::2] = sharpen_raw(B, factorB)
    proc_im[0::2, 0::2] = np.clip(G1, 0, 65535).astype(np.uint16)
    proc_im[1::2, 1::2] = np.clip(G2, 0, 65535).astype(np.uint16)

    return proc_im

def create_edge_mask(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img[:, :, 0]
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 3. Convert back to uint8 (absolute value)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # 4. Combine X and Y gradients
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    sobel_combined = cv2.GaussianBlur(sobel_combined, (3, 3), 0)

    return sobel_combined

def desaturate_mask(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsv[:, :, 1] = np.where(mask > 40, 50, hsv[:, :, 1])
    hsv[:, :, 1] = ( hsv[:, :, 1].astype(np.float32) * ((np.clip(170-mask.astype(np.float32), 80, 170))/170) ).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result

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

def reinhard_tone_map1(hdr_img, alpha=0.18):
    """
    Applies Reinhard Global Tone Mapping to an HDR image.
    
    Args:
        hdr_img: numpy array (H, W, 3) - Linear HDR image.
        alpha: Key value (middle gray) - usually 0.18.
        gamma: Gamma correction value.
        
    Returns:
        ldr_img: numpy array (H, W, 3) - Tone mapped LDR image [0, 1].
    """
    # 1. Compute Luminance (Y = 0.2126*R + 0.7152*G + 0.0722*B)
    # Using small epsilon to prevent log(0)
    luminance = 0.2126 * hdr_img[:,:,0] + 0.7152 * hdr_img[:,:,1] + 0.0722 * hdr_img[:,:,2]
    
    # 2. Compute Log-Average Luminance
    N = hdr_img.shape[0] * hdr_img.shape[1]
    log_mean_lum = np.exp(np.sum(np.log(luminance + 1e-10)) / N)
    
    # 3. Scale Luminance (Key adjustment)
    scaled_lum = (alpha / log_mean_lum) * luminance
    
    # 4. Reinhard Formula
    tone_mapped_lum = scaled_lum / (1 + scaled_lum)
    
    # 5. Map back to color channels and clamp
    # Avoid division by zero by using original luminance
    ratio = tone_mapped_lum / (luminance + 1e-10)
    
    # Apply ratio to each channel
    tonemapped_img = np.zeros_like(hdr_img)
    for i in range(3):
        tonemapped_img[:,:,i] = hdr_img[:,:,i] * ratio
        
    return np.clip(tonemapped_img*255.0, 0, 255).astype(np.uint8)

def desaturate(img, factor=0.5):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Scale saturation (0.0 = grayscale, 1.0 = original)
    s = np.clip(s * factor, 0, 255)
    
    # Merge and convert back
    hsv_new = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

def aces_tonemap(img):
    m1 = np.array([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ])
    # Output transform (back to sRGB/Rec.709)
    m2 = np.array([
        [1.60475, -0.53108, -0.07367],
        [-0.10208, 1.10813, -0.00605],
        [-0.00327, -0.07276, 1.07602]
    ])
    
    v = img @ m1.T
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    res = (a / b) @ m2.T
    
    return np.clip(res, 0.0, 1.0)

def hable_tonemap(x, exp=10):
    # Constants for John Hable's Uncharted 2 curve
    A = 0.15 #0.15 # Shoulder Strength
    B = 0.5 #0.50 # Linear Strength
    C = 5 #0.10 # Linear Angle
    D = 0.8 #0.20 # Toe Strength
    E = 0.02 #0.02 # Toe Numerator
    F = 0.8 #0.30 # Toe Denominator
    W = 1.5 #11.2 # Linear White Point Value

    def curve(x):
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B * C) + D * F)) - E / F

    # Apply the curve and normalize by the white point
    exposure_bias = exp
    curr = curve(exposure_bias * x)
    white_scale = 1.0 / curve(W)
    
    return curr * white_scale

def sigmoid_correct(x, A=0.2, B=10, C=-0.9, D=0.8):
    return ( A / (1 + np.exp(-B* (x+C))) ) + D

def main():
    #code
    file = args.input

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
        raw_im = raw.raw_image_visible.astype(np.uint16)

        raw_max = np.percentile(raw_im, 99)
        print("raw image scale top: ", raw_max)
        green_median = np.median(raw_im[::2, ::2])-black[0]
        

        #Apply black level subtraction
        proc_im1 = black_subtraction(raw_im, black[0], 65535)

        #Apply color scaling
        proc_im2 = color_scaling(proc_im1, wb)

        #proc_im3 = LCA_correction(proc_im2, 1.00005, 0.99995)
        proc_im3 = axial_CA_correction(proc_im2, 0.0, 0.0)

        dRG = RG_delta(proc_im3)
        dBG = BG_delta(proc_im3)

        raw.raw_image_visible[:] = proc_im1
        
        #Demosaicing
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, no_auto_scale=False, use_camera_wb=True, user_black=0, output_bps=16, 
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.raw)
        
    #print(rgb)

    #Apply color correction matrix
    rgb = apply_CCM(rgb, ccm)

    #Perform tone mapping to convert HDR to LDR
    rgb = rgb.astype(np.uint16)
    rgb_float = np.clip((rgb/65535.0), 0.0, 1.0).astype(np.float32)
    print(rgb_float.shape)

    tonemap = cv2.createTonemapReinhard(1,2,0,0.4) #gamma, intensity, light_adapt, color_adapt
    #tonemap = cv2.createTonemapDrago(gamma=1, saturation=0.7, bias=1)
    #tonemap = cv2.createTonemapMantiuk(gamma=2.2, scale=0.7, saturation=1.5)
    ldr = tonemap.process(rgb_float)
    #ldr = mobius_tonemap(rgb_float)

    exp = (250 / green_median) * 10
    print("green median: ", green_median)
    print("exp: ", exp)

    #ldr = hable_tonemap(rgb_float, 20)
    #ldr = ldr * sigmoid_correct(rgb_float)
    #print(np.max(ldr))
    ldr = ldr.astype(np.float32) / np.max(ldr)
    #ldr = aces_tonemap(rgb_float)
    ldr_8bit = np.clip(ldr * 255, 0, 255).astype('uint8')

    #ldr_8bit = reinhard_tone_map1(rgb_float)
    #ldr_8bit = desaturate(ldr_8bit, factor=0.5)

    ldr_8bit = apply_gamma(ldr_8bit, 1.5, 4.5)
    hsv = cv2.cvtColor(ldr_8bit, cv2.COLOR_BGR2HSV)
    edge_mask = create_edge_mask(ldr_8bit)
    #lower = np.array([120, 0, 0])
    #upper = np.array([190, 255, 255])
    #purple_mask= cv2.inRange(hsv, lower, upper)
    #purple_mask = cv2.GaussianBlur(purple_mask, (5, 5), 0)
    #purple_mask = np.where(purple_mask>1, 1, 0)
    #edge_mask = edge_mask.astype(np.float32) * (purple_mask.astype(np.float32))

    #ldr_8bit = desaturate_mask(ldr_8bit, edge_mask)
    #ldr_8bit = cv2.convertScaleAbs(ldr_8bit, alpha=1.2, beta=0)
    #ldr_8bit = sharpen(ldr_8bit, 0.2)
    

    #ldr_filtered = cv2.fastNlMeansDenoisingColored(ldr_8bit, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)
    #ldr_filtered = remove_axial_ca(ldr_8bit, 20, 0.5)

    #channel_max = np.max(ldr_8bit, axis=2)
    #channel_min = np.min(ldr_8bit, axis=2)
    #print(channel_max.shape)
    #saturation_mask = np.where(channel_max>253, 255, 0).astype(np.uint8)
    #saturation_mask = np.where(channel_min>253, 0, saturation_mask).astype(np.uint8)

    x = np.linspace(0, 1, 65535)
    y = hable_tonemap(x)

    # Plotting results
    plt.figure(1)
    #fig, ax = plt.subplots()
    #plt.imshow(dBG, cmap='gray', vmin=0, vmax=1000)
    #plt.imshow(raw_im, cmap='gray', vmin=black[0], vmax=np.max(raw_im))
    #plt.imshow((rgb_float*255).astype(np.uint8))
    plt.imshow(ldr_8bit)
    #plt.imshow(edge_mask, cmap="jet")
    #plt.imshow(saturation_mask)
    #plt.imshow(purple_mask, cmap="gray")
    #plt.imshow(hsv[:, :, 0], cmap="jet")
    #plt.colorbar()
    #plt.imshow(ldr_filtered)

    #plt.figure(2)
    #plt.plot(x, y)

    #plt.figure(3)
    #plt.hist(rgb_float.ravel(), 10000, [0.0002, 1], alpha=1)
    plt.show()
    #imageio.imsave('output_16bit.tiff', rgb)
    #imageio.imsave('output.jpg', ldr_8bit, quality=100)

    print("\nSaving Image Output...\n")
    input_filename = os.path.basename(file)
    input_filename = os.path.splitext(input_filename)[0]
    output_filenameL = f"output_rawL_{input_filename}.jpg"
    output_filenameR = f"output_rawR_{input_filename}.jpg"
    output_filenameLR = f"output_rawLR_{input_filename}.jpg"
    #print("Output Filename: ", output_filename)
    imageio.imsave(output_filenameR, ldr_8bit[:, int(ldr_8bit.shape[1]/2):, :], quality=100)
    imageio.imsave(output_filenameL, ldr_8bit[:, :int(ldr_8bit.shape[1]/2), :], quality=100)
    imageio.imsave(output_filenameLR, ldr_8bit, quality=100)


if __name__=="__main__":
    args = parse()
    main()