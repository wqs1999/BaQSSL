import pennylane as qml
import torch
from pennylane import numpy as np


class SimCirc14:
    def __init__(self, layers: int = 2,
                 sweeps_per_layer: int = 2,
                 activation_function=None):
        self.layers = layers
        self.sweeps = sweeps_per_layer
        self.activation = activation_function  # 需为PyTorch激活函数（如torch.sigmoid）
        self.n_qubits = None
        self.n_params = None

    def get_quantum_circuit(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_params = self.layers * self.sweeps * 4 * self.n_qubits  # 总参数数
        return self.circuit

    def circuit(self, inputs, params):  # 移除is_drawing参数
        idx = 0
        # 1. 输入编码（inputs：PyTorch Tensor，与后续流程兼容）
        qml.AngleEmbedding(features=inputs, wires=range(self.n_qubits), rotation='Y')

        for layer_no in range(self.layers):
            for _ in range(self.sweeps):
                # 第一旋转层（RY门）
                for i in range(self.n_qubits):
                    qml.RY(params[idx], wires=i)
                    idx += 1
                # 第一纠缠层（反向CRX）
                for i in reversed(range(self.n_qubits)):
                    target = (i + 1) % self.n_qubits
                    qml.CRX(params[idx], wires=[i, target])
                    idx += 1
                # 第二旋转层（RY门）
                for i in range(self.n_qubits):
                    qml.RY(params[idx], wires=i)
                    idx += 1
                # 第二纠缠层（间隔CRX）
                for i in range(self.n_qubits):
                    control = (i + self.n_qubits - 1) % self.n_qubits
                    target = (control + 3) % self.n_qubits
                    qml.CRX(params[idx], wires=[control, target])
                    idx += 1

            # 2. 层间激活（仅在实际运行时执行，绘制时因不计算测量值会自动跳过）
            if layer_no < self.layers - 1 and self.activation is not None:
                # 步骤1：获取测量结果（运行时为标量Tensor，绘制时为ExpectationMP对象）
                meas = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
                # 步骤2：将测量结果转为Python数值（绘制时不执行实际计算，此步自动跳过）
                meas_values = [m.item() for m in meas]
                # 步骤3：转为PyTorch Tensor并应用激活函数
                meas_tensor = torch.tensor(meas_values, dtype=torch.float32, device=inputs.device)
                activated_tensor = self.activation(meas_tensor)
                # 步骤4：转为数值列表，适配qml.RY的参数要求
                activated = activated_tensor.tolist()
                # 步骤5：激活后重新编码到量子态
                for i in range(self.n_qubits):
                    qml.RY(activated[i], wires=i)

        # 3. 最终测量
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    n_qubits = 4
    # 1. 初始化电路（PyTorch sigmoid激活）
    sim14 = SimCirc14(
        layers=2,
        sweeps_per_layer=2,
        activation_function=torch.sigmoid
    )
    circuit_func = sim14.get_quantum_circuit(n_qubits)
    print(f"电路总参数数：{sim14.n_params}")  # 预期：2×2×4×4=64

    # 2. 生成PyTorch Tensor输入和参数（CPU测试）
    inputs = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cpu")
    params = torch.rand(sim14.n_params, device="cpu")

    # 3. 绘制量子电路（无需is_drawing参数，绘制时自动不执行激活逻辑）
    print("\n=== 绘制量子电路 ===")
    drawn_circuit = qml.draw(circuit_func)(inputs, params)
    print(drawn_circuit)

    # 4. 运行量子电路（执行激活逻辑）
    # print("\n=== 运行量子电路 ===")
    # dev = qml.device("default.qubit", wires=n_qubits)
    # qnode = qml.QNode(circuit_func, dev)
    # meas_results = qnode(inputs, params)
    # # 转为PyTorch Tensor
    # meas_tensor = torch.tensor([m.item() for m in meas_results], device="cpu")
    # print(f"测量结果（各qubit的PauliZ期望值）：{meas_tensor}")
