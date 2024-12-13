from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def show_image_pixel():
    '''
    展示图像像素值
    :return:
    '''
    # 加载图片
    path = './Original_images/lenna.png'
    image = Image.open(path)
    image = image.convert('L')  # 转换为灰度图

    # 获取左上角32*32的子图
    sub_image = image.crop((0, 0, 8, 8))

    # 将像素值转换为0-255的范围并取整
    pixels = np.array(sub_image).astype(np.uint8)

    # 绘制表格
    fig, ax = plt.subplots()
    cax = ax.matshow(pixels, cmap='gray')  # 使用灰度色图

    # 在表格的每个格子中添加数字
    for (i, j), val in np.ndenumerate(pixels):
        ax.text(j, i, val, ha='center', va='center', color='white')

    plt.show()
    # 保存结果
    # plt.savefig('result.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def show_stego_compare():
    # show_image_pixel()
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Data preparation
    methods = ["DCT-based", "DWT-based", "DFT-based", "DeepLearning-based"]
    values = [25300, 16600, 8650, 26800]

    # Set Seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Create a figure with a specific size
    plt.figure(figsize=(8, 10))

    # Plot the bar chart
    plt.bar(methods, values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])

    # Add labels and title
    plt.xlabel("Steganography Methods", fontsize=14)
    plt.ylabel("Data Capacity (bytes)", fontsize=14)
    plt.title("Steganography Method Data Capacity Comparison", fontsize=18)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=14)

    # Set y-axis limits and add grid lines
    plt.ylim(0, 28000)
    plt.grid(axis='y', linestyle='--', alpha=0.9)

    # Add value labels on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 500, str(v), ha='center', va='bottom')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Display the chart
    plt.show()


if __name__ == '__main__':
    show_stego_compare()
