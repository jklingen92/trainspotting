import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt


def create_motion_kernel(start_point, end_point, kernel_size=65):
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    magnitude = np.sqrt(dx**2 + dy**2)
    
    dx = dx / magnitude * (kernel_size // 2)
    dy = dy / magnitude * (kernel_size // 2)
    
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    cv2.line(
        kernel,
        (center - int(dx), center - int(dy)),
        (center + int(dx), center + int(dy)),
        1,
        1
    )
    
    return kernel / np.sum(kernel)

def wiener_deblur_channel(channel, kernel, K=0.01):
    img_float = channel.astype(float) / 255
    img_fft = np.fft.fft2(img_float)
    kernel_fft = np.fft.fft2(kernel, s=img_float.shape)
    
    kernel_fft_conj = np.conj(kernel_fft)
    deblurred_fft = (kernel_fft_conj / (np.abs(kernel_fft)**2 + K)) * img_fft
    
    deblurred = np.real(np.fft.ifft2(deblurred_fft))
    deblurred = np.clip(deblurred, 0, 1)
    
    return (deblurred * 255).astype(np.uint8)

def wiener_deblur(image, kernel, K=0.01):
    # Split into channels and deblur each independently
    b, g, r = cv2.split(image)
    
    b_deblurred = wiener_deblur_channel(b, kernel, K)
    g_deblurred = wiener_deblur_channel(g, kernel, K)
    r_deblurred = wiener_deblur_channel(r, kernel, K)
    
    # Merge channels back
    return cv2.merge([b_deblurred, g_deblurred, r_deblurred])
