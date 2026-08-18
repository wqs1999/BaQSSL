import torch
import torch.nn as nn
import torch.nn.functional as F
import random, os, numpy as np
import pennylane as qml
from pennylane.qnn import TorchLayer
from CIRCUIT_HEAD import SimCirc14
from torch.amp import GradScaler, autocast

import Tool



def cb_quantum_circuit(inputs, weights, n_qubits):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    k = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    beta_q_raw = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    return k + [beta_q_raw]

class QuantumWrapper:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.__name__ = "cb_quantum_circuit_wrapper"

    def __call__(self, inputs, weights):
        return cb_quantum_circuit(inputs, weights, self.n_qubits)

class QuantumCBModule(nn.Module):
    def __init__(self, in_channels, n_qubits, device):
        super().__init__()
        self.C = in_channels
        self.n_qubits = n_qubits
        self.device = device

        self.global_pool = nn.AdaptiveAvgPool2d(1).to(device)
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
        nn.init.xavier_uniform_(self.proj_k.weight)

        self.beta_finetune = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.to(self.device)
        x_pool = self.global_pool(x)
        x_pool_flat = x_pool.view(B, C)
        q_input = self.q_input_proj(x_pool_flat)

        q_out = self.q_layer(q_input)
        k_quantum = q_out[:, :-1]
        k = self.proj_k(k_quantum)
        k = F.normalize(k, dim=1, eps=1e-6)

        beta_q_raw = q_out[:, -1:]
        beta_q = torch.sigmoid(beta_q_raw)
        beta_classic = self.beta_finetune(x_pool_flat)
        beta = 0.5 * beta_q + 0.5 * beta_classic

        k_expand = k.view(B, C, 1, 1)
        beta_expand = beta.view(B, 1, 1, 1)

        kT_x = torch.sum(k_expand * x, dim=1, keepdim=True)
        erased_x = x - beta_expand * k_expand * kT_x
        v = x
        replaced_x = erased_x + beta_expand * k_expand * v

        return replaced_x

def style_perturb_circuit(inputs, weights, n_qubits,
                          use_multi_rot=True,
                          use_crx_entangle=False):
    if use_multi_rot:
        rotation_types = ['X', 'Y', 'Z']
        for i in range(n_qubits):
            rot = rotation_types[i % 3]
            if rot == 'X':
                qml.RX(inputs[:, i], wires=i)
            elif rot == 'Y':
                qml.RY(inputs[:, i], wires=i)
            else:
                qml.RZ(inputs[:, i], wires=i)
    else:
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    if use_crx_entangle:
        style_qubits = [2, 3, 4, 5]
        for i in range(0, len(style_qubits), 2):
            control_q = style_qubits[i]
            target_q = style_qubits[i + 1]
            crx_angle = weights[0, control_q, 0].expand(inputs.shape[0])
            qml.CRX(crx_angle, wires=[control_q, target_q])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class PerturbCircuitWrapper:
    def __init__(self, n_qubits, use_multi_rot=True, use_crx_entangle=False):
        self.n_qubits = n_qubits
        self.use_multi_rot = use_multi_rot
        self.use_crx_entangle = use_crx_entangle
        self.__name__ = "style_perturb_circuit"

    def __call__(self, inputs, weights):
        return style_perturb_circuit(
            inputs, weights, self.n_qubits,
            use_multi_rot=self.use_multi_rot,
            use_crx_entangle=self.use_crx_entangle
        )

class PQCStylePerturb(nn.Module):
    def __init__(self, in_dim, n_qubits=6, n_perturb=3, device='cuda:0',
                 use_multi_rot=True,
                 use_crx_entangle=False,
                 use_adaptive_strength=True):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.n_perturb = n_perturb
        self.device = device
        self.use_multi_rot = use_multi_rot
        self.use_crx_entangle = use_crx_entangle
        self.use_adaptive_strength = use_adaptive_strength

        self.feat2qubit = nn.Linear(in_dim, n_qubits, bias=False).to(device)
        nn.init.xavier_uniform_(self.feat2qubit.weight)

        self.perturb_layers = nn.ModuleList()
        weight_shapes = {"weights": (3, n_qubits, 3)}

        for _ in range(n_perturb):
            circuit_wrapper = PerturbCircuitWrapper(
                n_qubits,
                use_multi_rot=self.use_multi_rot,
                use_crx_entangle=self.use_crx_entangle
            )
            qnode = qml.QNode(
                circuit_wrapper,
                qml.device("default.qubit", wires=n_qubits),
                diff_method="backprop"
            )
            pqc_layer = TorchLayer(qnode, weight_shapes).to(device)
            self.perturb_layers.append(pqc_layer)

        self.qubit2feat = nn.Linear(n_qubits, in_dim, bias=False).to(device)
        nn.init.constant_(self.qubit2feat.weight, 1.0 / n_qubits)

        if self.use_adaptive_strength:
            self.perturb_scale = nn.Parameter(torch.tensor(0.05, device=device), requires_grad=True)

    def forward(self, x):
        B = x.size(0)
        if len(x.shape) == 4:
            x_flat = x.view(B, -1)
            feat_dim = x_flat.size(1)
        else:
            x_flat = x
            feat_dim = self.in_dim

        q_input = self.feat2qubit(x_flat)

        perturb_feats = []
        for pqc_layer in self.perturb_layers:
            q_perturb = pqc_layer(q_input)
            a_prime = self.qubit2feat(q_perturb)

            if self.use_adaptive_strength:
                q_perturb_mean = torch.mean(q_perturb, dim=1, keepdim=True)
                adapt_strength = torch.tanh(q_perturb_mean) * self.perturb_scale
                a_prime = a_prime * adapt_strength
            else:
                a_prime = torch.tanh(a_prime) * 0.05

            perturb_feats.append(a_prime)

        x_expand = x_flat.unsqueeze(1)
        perturb_feats = torch.stack(perturb_feats, dim=1)
        all_feats = torch.cat([x_expand, perturb_feats], dim=1).view(-1, feat_dim)

        if len(x.shape) == 4:
            C, H, W = x.size(1), x.size(2), x.size(3)
            all_feats = all_feats.view(-1, C, H, W)

        return all_feats

def get_qlayer_output_dim(qlayer, input_dim, device):
    test_input = torch.randn(1, input_dim).to(device, dtype=torch.float32)
    with torch.no_grad():
        output = qlayer(test_input)
    return output.size(-1)

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
        x = self.pre_linear(x)
        if self.use_EA:
            x = x + self.ea(x) * self.eps
        z = self.q_layer(x)
        z = self.proj(z)
        return F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature=0.1, n_perturb=3):
    M = n_perturb + 1
    B_origin = z1.size(0) // M
    assert z1.size(0) == z2.size(0) == B_origin * M, "特征维度需匹配扩充后的样本数"

    z = torch.cat([z1, z2], dim=0)
    batch_size_total = z.size(0)

    sim_matrix = torch.mm(z, z.t()) / temperature
    sim_matrix = sim_matrix - torch.eye(batch_size_total, device=z.device) * 1e9

    pos_labels = torch.zeros(batch_size_total, dtype=torch.long, device=z.device)
    for b in range(B_origin):
        for p in range(M):
            idx1 = b * M + p
            idx2 = B_origin * M + b * M + p
            pos_labels[idx1] = idx2
    for b in range(B_origin):
        for p in range(M):
            idx2 = B_origin * M + b * M + p
            idx1 = b * M + p
            pos_labels[idx2] = idx1

    loss = F.cross_entropy(sim_matrix, pos_labels)
    return loss

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    input_size = 10
    n_perturb = 2
    n_qubits = 6

    TrainLoader, TestLoader, classes = Tool.DataLoader_1280(
        batch_size=128,
        num_workers=0,
        input_size=input_size,
        Train_num=640,
        Test_num=30,
        category=10
    )
    print(f"CIFAR-10类别：{classes}")

    qcnn = QCNN_Conv2d(in_channels=3, out_channels=5, kernel_size=2, input_size=input_size).to(device)
    cb_module = QuantumCBModule(in_channels=5, n_qubits=n_qubits, device=device).to(device)
    pool_layer = nn.MaxPool2d(kernel_size=2).to(device)

    with torch.no_grad():
        temp_input = torch.randn(1, 3, input_size, input_size).to(device)
        temp_out = qcnn(temp_input)
        temp_out_cb = cb_module(temp_out)
        temp_out_pool = pool_layer(temp_out_cb)
        classic_feat_dim = temp_out_pool.numel() // temp_out_pool.size(0)

    print(f"量子卷积输出维度：{temp_out.shape}")
    print(f"CB处理后维度：{temp_out_cb.shape}")
    print(f"池化后维度：{temp_out_pool.shape}")
    print(f"展平后经典特征维度：{classic_feat_dim}")

    pqc_perturb = PQCStylePerturb(
        in_dim=classic_feat_dim,
        n_qubits=n_qubits,
        n_perturb=n_perturb,
        device=device,
        use_multi_rot=True,
        use_crx_entangle=False,
        use_adaptive_strength=True
    ).to(device)

    with torch.no_grad():
        perturb_out = pqc_perturb(temp_out_pool)
    print(f"PQC风格扰动后维度（原始+{n_perturb}组扰动）：{perturb_out.shape}")

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

    Net = nn.Sequential(
        qcnn,
        cb_module,
        pool_layer,
        nn.BatchNorm2d(num_features=5).to(device),
        pqc_perturb,
        nn.Flatten(start_dim=1, end_dim=3),
        QuantumFeatHead(classic_feat_dim, q_input_dim, q_layer, device, use_EA=False, eps=0.1)
    ).to(device)

    print("\n模型结构（量子卷积+量子CB+量子风格扰动+量子特征头）：")
    print(Net)

    start_epoch = 0
    model_save_dir = "./Models_cifar10_cb_perturb_rgb_1"
    resume_model_path = os.path.join(model_save_dir, "model_27.pt")
    os.makedirs(model_save_dir, exist_ok=True)

    if os.path.exists(resume_model_path):
        Net = torch.load(resume_model_path, map_location=device)
        Net.to(device)
        start_epoch = 28
        print(f"\n已加载预训练模型，从epoch {start_epoch} 继续训练")
    else:
        print("\n未找到历史模型，从epoch 0 开始训练")

    quantum_params = []
    classic_params = []
    for name, param in Net.named_parameters():
        if "q_layer" in name or "perturb_layers" in name:
            quantum_params.append(param)
        else:
            classic_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": classic_params, "lr": 1e-3, "weight_decay": 1e-4},
        {"params": quantum_params, "lr": 5e-4, "weight_decay": 1e-5}
    ])

    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    train_loss_hist = []
    num_epochs = 90

    for epoch in range(start_epoch, num_epochs):
        Net.train()
        running_train_loss = 0.0
        for batch_idx, (views, _) in enumerate(TrainLoader):
            view1, view2 = views[0].to(device), views[1].to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
                z1 = Net(view1)
                z2 = Net(view2)
                loss = nt_xent_loss(z1, z2, temperature=0.1, n_perturb=n_perturb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * view1.size(0)



        epoch_train_loss = running_train_loss / len(TrainLoader.dataset)
        train_loss_hist.append(epoch_train_loss)

