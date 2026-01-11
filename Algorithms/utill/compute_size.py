import math

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


def calc_out_size(h_in, w_in, layers):
    """
    计算卷积 + 池化后的尺寸变化
    layers: list，每一层是一个dict：
        - type: "conv" 或 "pool"
        - kernel: 核大小 (int)
        - stride: 步长 (int)
        - pad: padding (int)（卷积层必填，池化层可不填）
    """
    h, w = h_in, w_in
    for i, layer in enumerate(layers, 1):
        if layer["type"] == "conv":
            k, s, p = layer["kernel"], layer["stride"], layer["pad"]
            h = math.floor((h + 2*p - k) / s) + 1
            w = math.floor((w + 2*p - k) / s) + 1
            # print(f"After Conv{i}: {h} x {w}")
        elif layer["type"] == "pool":
            k, s = layer["kernel"], layer["stride"]
            h = math.floor((h - k) / s) + 1
            w = math.floor((w - k) / s) + 1
            # print(f"After Pool{i}: {h} x {w}")
    return h, w


def cnn_output_size_example(kernel_size, stride, padding):
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


# cnn_output_size_example()