import torch.nn as nn
import torch
import torch.nn.functional as F
import pennylane as qml
import datetime
import matplotlib.pyplot as plt

torch.cuda.set_device(0)

# ========== 量子电路定义（仅保留随机 RY 电路） ==========
def circuit_Ran_Ry_4qubits(inputs, weights):
    """
    随机 RY 电路：4 量子比特，单次强纠缠层（7 个可训练参数）
    返回单个 PauliZ(1) 期望值（标量）
    """
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    qml.RY(weights[3], wires=3)
    qml.CRY(weights[4], wires=[1, 0])
    qml.CRY(weights[5], wires=[2, 1])
    qml.CRY(weights[6], wires=[1, 3])
    return qml.expval(qml.PauliZ(1))


# ========== 量子卷积层 ==========
class QCNN_Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, input_size: int = 32) -> None:
        super(QCNN_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_size = input_size
        self.dev = qml.device("default.qubit", wires=self.kernel_size ** 2)

        # 构建卷积核列表：out_channels × in_channels 个量子层
        self.kernels = []
        for i in range(self.out_channels):
            kernels_temp = []
            for j in range(self.in_channels):
                kernels_temp.append(self._create_filter())
            kernels_temp = nn.ModuleList(kernels_temp)
            self.kernels.append(kernels_temp)
        self.kernels = nn.ModuleList(self.kernels)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 保留但未使用

    def _create_filter(self):
        """统一创建随机 RY 量子滤波器"""
        weight_shapes = {"weights": (7,)}
        qnode = qml.QNode(circuit_Ran_Ry_4qubits, self.dev, diff_method="best")
        return qml.qnn.TorchLayer(qnode, weight_shapes)

    def init_out(self):
        """初始化输出缓存列表"""
        return [[] for _ in range(self.out_channels)]

    def CarryParallelKernel(self, kernels, inputs, index):
        """
        对每个输入通道并行应用对应的量子核，并对结果求和
        inputs: [batch, in_channels, 4] （4 = kernel_size²）
        """
        out = []
        for i in range(len(inputs[0])):          # 遍历 in_channels
            out.append(kernels[i](inputs[:, i:i+1, :]))  # 输出 [batch, 1]
        out = torch.stack(out, dim=1)             # [batch, in_channels, 1]
        out = torch.sum(out, dim=1)               # [batch, 1]
        self.out[index].append(out.squeeze(1))    # 存储 [batch]

    def forward(self, inputs):
        batch_size = len(inputs)
        in_channel_size = len(inputs[0])
        out_channel_size = self.out_channels
        new_h = inputs.size(2) - self.kernel_size + 1
        new_w = inputs.size(3) - self.kernel_size + 1

        self.out = self.init_out()   # 重置输出缓存

        # 滑动窗口
        for q in range(new_h):
            for p in range(new_w):
                # 提取 patch: [batch, C, k, k] -> [batch, C, k*k]
                patch = inputs[:, :, q:q+self.kernel_size, p:p+self.kernel_size]
                patch = patch.reshape(batch_size, in_channel_size, self.kernel_size ** 2)
                # 对每个输出通道执行并行卷积
                for idx in range(len(self.kernels)):
                    self.CarryParallelKernel(self.kernels[idx], patch, idx)

        # 拼接各通道输出
        output = []
        for idx in range(len(self.kernels)):
            # self.out[idx] 中每个元素是 [batch]，stack 后得到 [batch, 滑动窗口数]
            output.append(torch.stack(self.out[idx], dim=1))
        output = torch.stack(output, dim=1)  # [batch, out_channels, 滑动窗口数]

        # 重塑为特征图
        output = output.reshape(batch_size, out_channel_size, new_h, new_w)
        return output

