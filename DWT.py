import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
from utils import intToBit, bitToInt, wordToBit, bitToWord
import copy


def lsbVal(a, b):
    result = 0
    if a == b:
        result = 0
    elif a == 1 and b == 0:
        result = 2
    elif a == 0 and b == 1:
        result = - 2

    return result


def show_img_dwt(img_path, together = False, wavelet='haar'):
    '''
    显示图像的DWT变换效果
    :param img_path: 图像路径
    :param wavelet: 使用的小波基，默认为'haar'
    '''
    # 读取灰度图
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: 图像无法读取，请检查路径是否正确。")
        return

    # 进行DWT变换
    coeffs2 = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs2

    if together:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        images = [LL, LH, HL, HH]

        for ax, im in zip(axs.ravel(), images):
            ax.imshow(im, cmap='gray', vmin=np.min(images), vmax=np.max(images))
            ax.axis('off')  # 去掉坐标轴

        plt.subplots_adjust(wspace=0.05, hspace=0.1)  # 调整子图间距
        plt.tight_layout()  # 确保布局紧凑
        plt.show()
    else:
        # 显示原图
        plt.subplot(2, 2, 1)
        plt.imshow(LL, cmap='gray')
        plt.title('LL Subband')

        # 显示DWT的LL子带
        plt.subplot(2, 2, 2)
        plt.imshow(LH, cmap='gray')
        plt.title('LH Subband')

        # 显示DWT的LH子带
        plt.subplot(2, 2, 3)
        plt.imshow(HL, cmap='gray')
        plt.title('HL Subband')

        # 显示DWT的HL子带
        plt.subplot(2, 2, 4)
        plt.imshow(HH, cmap='gray')
        plt.title('HH Subband')

        plt.tight_layout()
        plt.show()


class DWT:
    def __init__(self, wavelet='haar', end_tag="*"):
        """
        初始化 DWT Steganography 类。

        :param wavelet: 使用的小波基，默认为 'haar'
        :param end_tag: 消息结束标志，默认为 '*'
        """
        self.wavelet = wavelet
        self.end_tag = end_tag

    def dwtEncode(self, img_path, msg, is_show=False):
        """
        Embed a message into an image using DWT and LSB modification.

        :param img_path: Path to the input image.
        :param msg: The message to embed.
        :param is_show: Whether to display the original and encoded images.
        :return: The modified image with the embedded message.
        """
        img = cv2.imread(img_path)
        blue, green, red = cv2.split(img)  # pada opencv menggunakan format b, g, r
        msg = msg + self.end_tag
        # Proses merubah pesan String to Bit
        bitMessage = wordToBit(msg)

        # mendapatkan panjang pesan
        bitLenght = len(bitMessage)
        index = 0

        # Proses DWT-2D Red
        coeffsr = pywt.dwt2(red, self.wavelet)
        cAr, (cHr, cVr, cDr) = coeffsr
        # print (cAr)

        # Proses DWT-2D Green
        coeffsg = pywt.dwt2(green, self.wavelet)
        cAg, (cHg, cVg, cDg) = coeffsg

        # Proses DWT-2D Blue
        coeffsb = pywt.dwt2(blue, self.wavelet)
        cAb, (cHb, cVb, cDb) = coeffsb

        # inisialisasi cA baru tempat pesan akan disimpan
        cArResult = copy.deepcopy(cAr)
        cAgResult = copy.deepcopy(cAg)
        cAbResult = copy.deepcopy(cAb)

        # Proses menyisipkan pesan ke dalam gambar
        for i in range(len(cAr)):
            for j in range(len(cAr)):
                # red
                if index < bitLenght:
                    lsbPixel = intToBit(int(cAr[i, j]))[-2]
                    cArResult[i, j] = cAr[i, j] + lsbVal(bitMessage[index], lsbPixel)
                    index += 1
                # green
                if index < bitLenght:
                    lsbPixel = intToBit(int(cAg[i, j]))[-2]
                    cAgResult[i, j] = cAg[i, j] + lsbVal(bitMessage[index], lsbPixel)
                    index += 1
                # blue
                if index < bitLenght:
                    lsbPixel = intToBit(int(cAb[i, j]))[-2]
                    cAbResult[i, j] = cAb[i, j] + lsbVal(bitMessage[index], lsbPixel)
                    index += 1

        # convert dengan IDWT Red
        # print(cArResult)
        coeffsr2 = cArResult, (cHr, cVr, cDr)
        idwr = pywt.idwt2(coeffsr2, self.wavelet)
        idwr = np.uint8(idwr)

        # convert dengan IDWT Green
        coeffsg2 = cAgResult, (cHg, cVg, cDg)
        idwg = pywt.idwt2(coeffsg2, self.wavelet)
        idwg = np.uint8(idwg)

        # convert dengan IDWT Blue
        coeffsb2 = cAbResult, (cHb, cVb, cDb)
        idwb = pywt.idwt2(coeffsb2, self.wavelet)
        idwb = np.uint8(idwb)

        ImageResult = cv2.merge((idwb, idwg, idwr))

        if is_show:
            fig, axs = plt.subplots(1, 2)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[0].imshow(img)
            axs[0].set_title('Original Image')
            # axs[0].axis('off')  # 关闭坐标轴

            axs[1].imshow(cv2.cvtColor(ImageResult, cv2.COLOR_BGR2RGB))
            # axs[1].axis('off')  # 关闭坐标轴
            axs[1].set_title('Encoded Image')
            plt.figtext(0.5, 0.1, 'DWT Steganography', ha='center', va='center', fontsize=12)

            plt.tight_layout()
            plt.show()

        return ImageResult

    def dwtDecode(self, img):
        """
        Extract the hidden message from an image using DWT and LSB extraction.

        :param img: The image containing the hidden message.
        :return: The extracted message.
        """
        # pada opencv menggunakan format b, g, r
        blue, green, red = cv2.split(img)

        # Proses DWT-2D Red
        coeffsr = pywt.dwt2(red, self.wavelet)
        cAr, (cHr, cVr, cDr) = coeffsr

        # Proses DWT-2D Green
        coeffsg = pywt.dwt2(green, self.wavelet)
        cAg, (cHg, cVg, cDg) = coeffsg

        # Proses DWT-2D Blue
        coeffsb = pywt.dwt2(blue, self.wavelet)
        cAb, (cHb, cVb, cDb) = coeffsb
        bit = []

        for i in range(len(cAr)):
            for j in range(len(cAr)):
                if len(intToBit(int(cAr[i, j]))) > 2:
                    bit.append(intToBit(int(cAr[i, j]))[-2])
                else:
                    bit.append('0')

                if len(intToBit(int(cAg[i, j]))) > 2:
                    bit.append(intToBit(int(cAg[i, j]))[-2])
                else:
                    bit.append('0')

                if len(intToBit(int(cAb[i, j]))) > 2:
                    bit.append(intToBit(int(cAb[i, j]))[-2])
                else:
                    bit.append('0')

        return bitToWord(bit)



if __name__ == '__main__':
    # path = './Original_images/lenna.png'
    path = './Original_images/sea.png'
    # path = './Original_images/pepper.png'
    save_dir = './result/DWT_images/'
    os.makedirs(save_dir, exist_ok=True)
    show_img_dwt(path,True)

    msg = 'HelloWorld'

    # ==================== DWT ======================
    # ================= Encode ======================
    dwt = DWT()
    ImageResult = dwt.dwtEncode(path, msg, True)
    file_name = os.path.basename(path)
    save_path = save_dir + file_name[:-4] + "_encoded.png"

    cv2.imwrite(save_path, ImageResult)
    # ================= Decode ======================
    img2 = cv2.imread(save_path)
    msgResult = dwt.dwtDecode(img2)
    print(msgResult)
