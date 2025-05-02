import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
import cv2

# Загрузка изображения
image_path = "src/20210103_161604.jpg"
image = np.array(Image.open(image_path))

def showOrigImg(image):
    print("Размер изображения:", image.shape)
    plt.imshow(image)
    plt.title("Исходное изображение")
    plt.show()

#showOrigImg(image)

def build_autoencoder():
    inputs = Input(shape=(None, None, 3))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Middle
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(inputs, decoded)

def prepare_image(img, target_multiple=32):
    """Подгоняем размер изображения к кратному target_multiple"""
    h, w = img.shape[:2]
    new_h = h - h % target_multiple
    new_w = w - w % target_multiple
    return img[:new_h, :new_w]

# Подготовка данных
processed_img = prepare_image(image) / 255.0
input_img = np.expand_dims(processed_img, axis=0)

# Создание модели
autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Обучение
history = autoencoder.fit(input_img, input_img, epochs=20, batch_size=1)

# Визуализация процесса обучения
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Сжатие и восстановление
compressed = autoencoder.predict(input_img)[0]
compressed_image = (compressed * 255).astype(np.uint8)


### Архитектура
#from tensorflow.keras.utils import plot_model
#plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True, show_layer_names=True)

#from tensorflow.keras.utils import model_to_dot
#from IPython.display import SVG

#SVG(model_to_dot(autoencoder, show_shapes=True).create(prog='dot', format='svg'))
###

# Расчет метрик
def calculate_metrics(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(255 / rmse)

    from skimage.metrics import structural_similarity as ssim
    image_gray=cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    restored_gray=cv2.cvtColor(compressed.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    ssim_value, _=ssim(image_gray,restored_gray,full=True)
    return mse, rmse, psnr, ssim_value

mse_ae, rmse_ae, psnr_ae, ssim_ae = calculate_metrics(processed_img*255, compressed_image)
print(f"Autoencoder Compression - MSE: {mse_ae:.2f}, RMSE: {rmse_ae:.2f}, PSNR: {psnr_ae:.2f} dB, SSIM: {ssim_ae:.2f}")

# Визуализация результатов
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Оригинал")
plt.subplot(1, 3, 2)
plt.imshow(compressed_image)
plt.title(f"Сжатое (PSNR: {psnr_ae:.2f} dB)")
plt.subplot(1, 3, 3)
plt.imshow(np.abs(processed_img*255 - compressed_image), cmap='hot')
plt.title("Разница")
plt.colorbar()
plt.show()



