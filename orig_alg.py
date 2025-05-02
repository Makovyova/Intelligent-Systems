import cv2
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Загрузка изображения (из папки src, тут же)
image_path = "src/20210103_161604.jpg"
image = np.array(Image.open(image_path))

def showOrigImg(image):
  print("Размер изображения:", image.shape)
  plt.imshow(image)
  plt.title("Исходное изображение")
  plt.show()

def rgb_to_ycbcr(image):
    y = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    cb = -0.1687 * image[:,:,0] - 0.3313 * image[:,:,1] + 0.5 * image[:,:,2] + 128
    cr = 0.5 * image[:,:,0] - 0.4187 * image[:,:,1] - 0.0813 * image[:,:,2] + 128
    return np.stack([y, cb, cr], axis=2)

def quantize(channel, step):
    return (channel // step) * step + step / 2

def downsample_420(channel):
    return channel[::2, ::2]  # Берём каждый 2-й пиксель по вертикали и горизонтали
'''
def subsample_420(channel):
    return (channel[::2, ::2] + channel[1::2, ::2] + channel[::2, 1::2] + channel[1::2, 1::2]) / 4

cb_subsampled = subsample_420(ycbcr[..., 1])
cr_subsampled = subsample_420(ycbcr[..., 2])
'''

def upsample_420(channel, original_shape):
    # Линейная интерполяция для увеличения в 2 раза
    from scipy.ndimage import zoom
    return zoom(channel, zoom=(original_shape[0]/channel.shape[0],
                                original_shape[1]/channel.shape[1]))

def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)

def calculate_metrics(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(255 / rmse)

    from skimage.metrics import structural_similarity as ssim
    image_gray=cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    restored_gray=cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    ssim_value, _=ssim(image_gray,restored_gray,full=True)
    return mse, rmse, psnr, ssim_value

showOrigImg(image)

ycbcr = rgb_to_ycbcr(image)

step_y = 4  # Шаг квантования для Y
step_c = 8  # Шаг квантования для Cb/Cr (больше = сильнее сжатие)

# Уменьшаем Cb и Cr в 2 раза (4:2:0 — на 4 пикселя Y приходится 1 Cb и 1 Cr)
cb_subsampled = downsample_420(ycbcr[:,:,1])  # Cb
cr_subsampled = downsample_420(ycbcr[:,:,2])  # Cr

y_quantized = quantize(ycbcr[:,:,0], step_y)
cb_quantized = quantize(cb_subsampled, step_c)
cr_quantized = quantize(cr_subsampled, step_c)

cb_restored = upsample_420(cb_quantized, ycbcr.shape[:2])
cr_restored = upsample_420(cr_quantized, ycbcr.shape[:2])

restored_image = ycbcr_to_rgb(y_quantized, cb_restored, cr_restored)

mse, rmse, psnr, ssim_value = calculate_metrics(image, restored_image)
print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, PSNR: {psnr:.2f} dB, SSIM: {ssim_value}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Оригинал")
plt.subplot(1, 2, 2)
plt.imshow(restored_image)
plt.title(f"Сжатое (PSNR: {psnr:.2f} dB)")
plt.show()

# С разницей
plt.figure(figsize=(18, 10))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Оригинал")
plt.subplot(1, 3, 2)
plt.imshow(restored_image)
plt.title(f"Сжатое (PSNR: {psnr:.2f} dB)")
plt.subplot(1, 3, 3)
plt.imshow(np.abs(image-restored_image), cmap='hot')
plt.title("Разница")
plt.colorbar()
plt.show()

'''
from skimage.metrics import structural_similarity as ssim
image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
restored_gray=cv2.cvtColor(restored_image, cv2.COLOR_BGR2GRAY)
ssim_value, _=ssim(image_gray,restored_gray,full=True)
#ssim_value=ssim(image, restored_image, multichannel=True, data_range=255)
#print(f'SSIM: {ssim_value}')
'''