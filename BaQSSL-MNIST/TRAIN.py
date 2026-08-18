import torch
import torch.nn as nn
import torch.nn.functional as F
import random, os, numpy as np
import pennylane as qml
from pennylane.qnn import TorchLayer
from CIRCUIT_HEAD import SimCirc14
import time
import Tool




class QuantumCBModule(nn.Module):
    def __init__(self, in_channels, n_qubits, device):
        super().__init__()
        self.C = in_channels
        self.n_qubits = n_qubits
        self.device = device

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.q_input_proj = nn.Linear(in_channels, n_qubits, bias=False).to(device)
        nn.init.xavier_uniform_(self.q_input_proj.weight)

        circuit = QuantumWrapper(n_qubits)
        weight_shapes = {"weights": (4, n_qubits, 3)}
        qnode = qml.QNode(
            circuit,
            qml.device("default.qubit", wires=n_qubits),
            diff_method="backprop"
        )
        self.q_layer = TorchLayer(qnode, weight_shapes).to(device)

        self.proj_k = nn.Linear(n_qubits, in_channels).to(device)
        nn.init.constant_(self.proj_k.weight, 1.0 / n_qubits)

        self.beta_finetune = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        B, C, H, W = x.shape

        x_pool = self.global_pool(x)
        x_pool_flat = x_pool.view(B, C)
        q_input = self.q_input_proj(x_pool_flat)
        q_out = self.q_layer(q_input)

        k = self.proj_k(q_out[:, :self.n_qubits])
        beta_q_raw = q_out[:, -1:]
        beta_q = torch.sigmoid(beta_q_raw)
        beta_classic = self.beta_finetune(x_pool_flat)
        beta = 0.5 * beta_q + 0.5 * beta_classic

        k_expand = k.view(B, C, 1, 1)
        beta_expand = beta.view(B, 1, 1, 1)
        kT_x = torch.sum(k_expand * x, dim=1, keepdim=True)
        erased_x = x - beta_expand * k_expand * kT_x
        replaced_x = erased_x + beta_expand * k_expand * x

        return replaced_x


class QuantumFeatHead(nn.Module):
    def __init__(self, classic_in_dim, q_input_dim, qlayer, device, use_EA=True, eps=0.1):
        super().__init__()
        self.use_EA = use_EA
        self.eps = eps
        self.device = device
        self.pre_linear = nn.Linear(classic_in_dim, q_input_dim).to(device)
        self.ea = nn.Linear(q_input_dim, q_input_dim, bias=False).to(device)
        nn.init.zeros_(self.ea.weight)
        self.q_layer = qlayer.to(device)
        self.q_output_dim = get_qlayer_output_dim(qlayer, q_input_dim, device)
        print(f"量子层输入维度：{q_input_dim}, 量子层输出维度：{self.q_output_dim}")
        self.proj = nn.Sequential(
            nn.Linear(self.q_output_dim, 128).to(device),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128).to(device)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.pre_linear(x)
        if self.use_EA:
            x = x + self.ea(x) * self.eps
        z = self.q_layer(x)
        z = self.proj(z)
        return nn.functional.normalize(z, dim=1)


def nt_xent_loss(z1, z2, temperature=0.1):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    labels = torch.arange(B, device=z.device)
    pos_idx = torch.cat([labels + B, labels])
    sim = sim - 1e9 * torch.eye(2 * B, device=z.device)
    loss = F.cross_entropy(sim, pos_idx)
    return loss


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_size = 10

    TrainLoader, TestLoader, classes = Tool.DataLoader_1280(
        batch_size=128,
        num_workers=0,
        input_size=input_size,
        Train_num=50,
        Test_num=30,
        category=10
    )
    print(f"MNIST类别：{classes}")

    n_qubits = 6

    temp_qcnn = QCNN(in_channels=1, out_channels=5, kernel_size=2, input_size=input_size).to(device)
    temp_cb = QuantumCBModule(in_channels=5, n_qubits=n_qubits, device=device).to(device)

    temp_input = torch.randn(1, 1, input_size, input_size).to(device)
    temp_output = temp_qcnn(temp_input)
    temp_output_cb = temp_cb(temp_output)
    temp_output_mix = temp_output_cb
    temp_pool = nn.MaxPool2d(kernel_size=2)(temp_output_mix)
    classic_feat_dim = temp_pool.numel() // temp_pool.size(0)

    sim14 = SimCirc14(layers=3, sweeps_per_layer=2)
    circ_func = sim14.get_quantum_circuit(n_qubits)
    if not hasattr(circ_func, "__name__"):
        circ_func.__name__ = "sim14_quantum_circuit"
    qnode = qml.QNode(
        circ_func,
        qml.device("default.qubit", wires=n_qubits),
        diff_method="best"
    )
    weight_shapes = {"params": sim14.n_params}
    q_layer = TorchLayer(qnode, weight_shapes).to(device)
    q_input_dim = n_qubits

    Net = torch.nn.Sequential(
        QCNN(in_channels=1, out_channels=5, kernel_size=2, input_size=input_size).to(device),
        temp_cb,
        torch.nn.MaxPool2d(kernel_size=2).to(device),
        torch.nn.BatchNorm2d(num_features=5).to(device),
        torch.nn.Flatten(start_dim=1, end_dim=3),
        QuantumFeatHead(classic_feat_dim, q_input_dim, q_layer, device, use_EA=False, eps=0.1)
    ).to(device)


    print(Net)

    quantum_params = []
    classic_params = []
    for name, param in Net.named_parameters():
        if "q_layer" in name:
            quantum_params.append(param)
        else:
            classic_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": classic_params, "lr": 5e-4},
        {"params": quantum_params, "lr": 5e-4}
    ], weight_decay=1e-5)

    start_epoch = 0
    train_loss_hist = []
    for epoch in range(start_epoch, 190):
        epoch_start_time = time.time()
        Net.train()
        running_train_loss = 0.0
        for batch_idx, (views, _) in enumerate(TrainLoader):
            view1, view2 = views[0].to(device), views[1].to(device)
            optimizer.zero_grad()

            z1 = Net[0](view1)
            z1 = Net[1](z1)
            z1 = Net[2](z1, force_apply=(torch.rand(1).item() <= 0.6))
            z1 = Net[3:](z1)

            z2 = Net[0](view2)
            z2 = Net[1](z2)
            z2 = Net[2](z2, force_apply=(torch.rand(1).item() <= 0.6))
            z2 = Net[3:](z2)

            loss = nt_xent_loss(z1, z2, temperature=0.1)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * view1.size(0)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_train_loss = running_train_loss / len(TrainLoader.dataset)
        train_loss_hist.append(epoch_train_loss)
        print(f"Epoch {epoch+1}/{190}, Loss: {epoch_train_loss:.4f}, Time: {epoch_duration:.1f}s")
