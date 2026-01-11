def compute_output_size(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    计算卷积或池化操作后的输出尺寸（单层）
    参数：
        h_w: 输入尺寸 (height/width), 如 23
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 空洞卷积的膨胀系数（默认为1）
    返回：
        输出尺寸 (整数)
    """
    return ((h_w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def cnn_output_size_example():
    # 输入尺寸：23 x 23
    h, w = 23, 23

    # conv1: kernel_size=3, stride=1, padding=1
    h = compute_output_size(h, kernel_size=3, stride=1, padding=1)
    w = compute_output_size(w, kernel_size=3, stride=1, padding=1)
    # maxpool1: kernel_size=2, stride=2 (默认)
    h = compute_output_size(h, kernel_size=2, stride=2)
    w = compute_output_size(w, kernel_size=2, stride=2)

    # conv2: kernel_size=3, stride=1, padding=1
    h = compute_output_size(h, kernel_size=3, stride=1, padding=1)
    w = compute_output_size(w, kernel_size=3, stride=1, padding=1)
    # maxpool2: kernel_size=2, stride=2
    h = compute_output_size(h, kernel_size=2, stride=2)
    w = compute_output_size(w, kernel_size=2, stride=2)

    print(f"Final feature map size: {h} x {w}")
    print(f"Flattened vector length: {128 * h * w}")  # conv2输出通道数是128


cnn_output_size_example()