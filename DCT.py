import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from PIL import Image

# 定义量化矩阵
quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def show_img_dct(img_path):
    '''
    显示图像的DCT变换
    :param img_path: 图像路径
    '''
    # 读取灰度图
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: 图像无法读取，请检查路径是否正确。")
        return

    # 进行DCT变换
    image_dct = cv2.dct(np.float32(image))

    # 显示原图
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(abs(image_dct)), cmap='gray')
    plt.title('DCT Coefficients (log-scale)')
    plt.show()


def show_blocks_dct(image_path):
    '''
    显示图像的8x8分块DCT变换
    :param image_path: 图像路径
    '''
    # 读取灰度图
    gray_image = cv2.imread(image_path, 0)

    # 对灰度通道进行8x8分块；并进行DCT和量化处理
    h, w = gray_image.shape[:2]
    blocks_dct = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = np.float32(gray_image[i:i + 8, j:j + 8] - 128)
            block_dct = cv2.dct(block)
            blocks_dct[i:i + 8, j:j + 8] = block_dct

    # 显示
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(blocks_dct, cmap='gray')
    plt.title('DCT Coefficients (8x8 blocks)')
    plt.show()


def show_one_blocks_dct(image_path):
    '''
    显示图像的最左上角的8x8分块DCT系数和量化后的DCT系数
    :param image_path: 图像路径
    '''
    # 读取灰度图
    gray_image = cv2.imread(image_path, 0)

    # 对灰度通道进行8x8分块；并进行DCT和量化处理
    h, w = gray_image.shape[:2]
    blocks_dct = np.zeros((h, w), dtype=np.float32)
    blocks_quantized_dct = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = np.float32(gray_image[i:i + 8, j:j + 8] - 128)
            block_dct = cv2.dct(block)
            blocks_dct[i:i + 8, j:j + 8] = block_dct

            # 量化DCT系数
            block_quantized_dct = block_dct / quantization_matrix
            # block_quantized_dct = np.round(block_dct / quantization_matrix + 0.5)
            blocks_quantized_dct[i:i + 8, j:j + 8] = block_quantized_dct

    # 显示
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.log(abs(blocks_dct[0:8, 0:8])), cmap='gray')
    # plt.colorbar()
    plt.title('DCT Coefficients')

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(abs(blocks_quantized_dct[0:8, 0:8])), cmap='gray')
    # plt.colorbar()
    plt.title('Quantized DCT Coefficients')
    plt.show()
    # 逆量化
    # inverse = blocks_quantized_dct[0:8, 0:8] / quantization_matrix


class DCT():
    def __init__(self):  # Constructor
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0

    def dctEncoder(self, img, secret_msg):
        secret = secret_msg
        self.message = str(len(secret)) + '*' + secret
        self.bitMess = self.toBits()
        # get size of image in pixels
        row, col = img.shape[:2]
        self.oriRow, self.oriCol = row, col
        if ((col / 8) * (row / 8) < len(secret)):
            print("Error: Message too large to encode in image")
            return False
        # make divisible by 8x8
        if row % 8 != 0 or col % 8 != 0:
            img = self.addPadd(img, row, col)

        row, col = img.shape[:2]
        # split image into RGB channels
        bImg, gImg, rImg = cv2.split(img)
        # message to be hid in blue channel so converted to type float32 for dct function
        bImg = np.float32(bImg)
        # break into 8x8 blocks
        imgBlocks = [np.round(bImg[j:j + 8, i:i + 8] - 128) for (j, i) in itertools.product(range(0, row, 8),
                                                                                            range(0, col, 8))]
        # Blocks are run through DCT function
        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
        # blocks then run through quantization table
        quantizedDCT = [np.round(dct_Block / quantization_matrix) for dct_Block in dctBlocks]
        # set LSB in DC value corresponding bit of message
        messIndex = 0
        letterIndex = 0
        for quantizedBlock in quantizedDCT:
            # find LSB in DC coeff and replace with message bit
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            DC[7] = self.bitMess[messIndex][letterIndex]
            DC = np.packbits(DC)
            DC = np.float32(DC)
            DC = DC - 255
            quantizedBlock[0][0] = DC
            letterIndex = letterIndex + 1
            if letterIndex == 8:
                letterIndex = 0
                messIndex = messIndex + 1
                if messIndex == len(self.message):
                    break
        # blocks run inversely through quantization table
        sImgBlocks = [quantizedBlock * quantization_matrix + 128 for quantizedBlock in quantizedDCT]
        # blocks run through inverse DCT
        # sImgBlocks = [cv2.idct(B)+128 for B in quantizedDCT]
        # puts the new image back together
        sImg = []
        for chunkRowBlocks in self.chunks(sImgBlocks, col / 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        # converted from type float32
        sImg = np.uint8(sImg)
        # show(sImg)
        sImg = cv2.merge((sImg, gImg, rImg))
        return sImg

    def dctDecoder(self, img):
        row, col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0
        # split image into RGB channels
        bImg, gImg, rImg = cv2.split(img)
        # message hid in blue channel so converted to type float32 for dct function
        bImg = np.float32(bImg)
        # break into 8x8 blocks
        imgBlocks = [bImg[j:j + 8, i:i + 8] - 128 for (j, i) in itertools.product(range(0, row, 8),
                                                                                  range(0, col, 8))]
        # blocks run through quantization table
        # quantizedDCT = [dct_Block/ (quant) for dct_Block in dctBlocks]
        quantizedDCT = [img_Block / quantization_matrix for img_Block in imgBlocks]
        i = 0
        # message extracted from LSB of DC coeff
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            if DC[7] == 1:
                buff += (0 & 1) << (7 - i)
            elif DC[7] == 0:
                buff += (1 & 1) << (7 - i)
            i = 1 + i
            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i = 0
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                    except:
                        pass
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize)) + 1:]
        # blocks run inversely through quantization table
        sImgBlocks = [quantizedBlock * quantization_matrix + 128 for quantizedBlock in quantizedDCT]
        # blocks run through inverse DCT
        # sImgBlocks = [cv2.idct(B)+128 for B in quantizedDCT]
        # puts the new image back together
        sImg = []
        for chunkRowBlocks in self.chunks(sImgBlocks, col / 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        # converted from type float32
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg, gImg, rImg))
        ##sImg.save(img)
        # dct_decoded_image_file = "dct_" + original_image_file
        # cv2.imwrite(dct_decoded_image_file,sImg)
        return ''

    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]

    def addPadd(self, img, row, col):
        img = cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))
        return img

    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8, '0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8, '0')
        return bits


def compare_png_jpg_compression(png_path, jpg_quality=95):
    """
    将PNG图像转换为JPEG格式并进行压缩，然后比较压缩前后的图像。

    :param png_path: 输入的PNG图像路径
    :param jpg_quality: JPEG压缩质量 (0-100)
    """
    # 读取原始PNG图像
    original_image = Image.open(png_path)
    # 创建一个临时的JPEG文件名
    jpg_path = png_path.replace('.png', '_compressed.jpg')

    # 将PNG图像保存为JPEG格式，并指定压缩质量
    original_image.save(jpg_path, 'JPEG', quality=jpg_quality)

    # 读取压缩后的JPEG图像
    compressed_image = Image.open(jpg_path)
    # 转换为numpy数组以便于matplotlib显示
    original_np = np.array(original_image)
    compressed_np = np.array(compressed_image)

    # 显示原始和压缩后的图像
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(original_np)
    axs[0].set_title('Original PNG Image')
    axs[0].axis('off')  # 关闭坐标轴

    axs[1].imshow(compressed_np)
    axs[1].set_title(f'Compressed JPEG Image (Quality {jpg_quality})')
    axs[1].axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.show()


def test_dct_encoder(path, save_path, plt_show=False):
    '''
    测试DCT encoder()
    :param path:
    :param save_path:
    :return:
    '''
    dct_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # secret_msg = input("Enter the message you want to hide: ")
    secret_msg = "HelloWorld"
    print("The message length is: ", len(secret_msg))
    dct_img_encoded = DCT().dctEncoder(dct_img, secret_msg)
    cv2.imwrite(save_path, dct_img_encoded)
    # cv2.imshow("DCT Encoded Image", dct_img_encoded)
    # cv2.waitKey(0)

    if plt_show:
        # 将BGR图像转换为RGB
        dct_img_rgb = cv2.cvtColor(dct_img, cv2.COLOR_BGR2RGB)
        dct_img_encoded_rgb = cv2.cvtColor(dct_img_encoded, cv2.COLOR_BGR2RGB)
        # 创建一个图形和两个子图
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # 在第一个子图中显示原始图像
        axs[0].imshow(dct_img_rgb)
        axs[0].set_title('Original Image')
        axs[0].axis('off')  # 关闭坐标轴

        # 在第二个子图中显示编码后的图像
        axs[1].imshow(dct_img_encoded_rgb)
        axs[1].set_title('Encoded Image')
        axs[1].axis('off')  # 关闭坐标轴

        plt.figtext(0.5, 0.05, 'DCT Steganography', ha='center', va='center', fontsize=12)
        # 显示图形
        plt.show()

    return dct_img_encoded


if __name__ == '__main__':
    # path = './Original_images/lenna.png'
    path = './Original_images/sea.png'
    # path = './Original_images/pepper.png'
    save_dir = './result/DCT_images/'
    os.makedirs(save_dir, exist_ok=True)

    # show_img_dct(path) # 显示整张图的dct系数分布
    # show_blocks_dct(path) # 显示图像的分块dct系数分布
    # show_one_blocks_dct(path) # 显示图像的第一个分块的dct系数分布
    # 根据path提取文件名
    file_name = os.path.basename(path)
    save_path = save_dir + file_name[:-4] + "_encoded.png"
    # 测试DCT隐写和提取密文
    dct_encoder_image = test_dct_encoder(path, save_path, plt_show=True)
    dct_hidden_text = DCT().dctDecoder(dct_encoder_image)
    print("-" * 50)
    print("The hidden message is: ", dct_hidden_text)

    # 测试比较JPEG压缩前后的区别
    # compare_png_jpg_compression(path, 30)
