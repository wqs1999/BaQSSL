import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tv_transforms
from torch.utils.data import Subset
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import pennylane as qml
from pennylane.qnn import TorchLayer
from CIRCUIT_HEAD import SimCirc14
from QCNN import MQCNN_Conv2d

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cb_quantum_circuit(inputs, weights, n_qubits):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    k = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    beta_q_raw = qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    return k + [beta_q_raw]

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
        if n_qubits == 8:
            pairs = [[4,5], [6,7]]
        elif n_qubits ==6:
            pairs = [[2,3], [4,5]]
        else:
            pairs = []
        for ctrl, tgt in pairs:
            angle = weights[0, ctrl, 0]
            qml.CRX(angle, wires=[ctrl, tgt])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumWrapper:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.__name__ = "cb_quantum_circuit_wrapper"
    def __call__(self, inputs, weights):
        return cb_quantum_circuit(inputs, weights, self.n_qubits)

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
    def __init__(self, in_dim, n_qubits=6, device='cuda:0',
                 use_multi_rot=True, use_crx_entangle=False, use_adaptive_strength=True):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.device = device
        self.use_multi_rot = use_multi_rot
        self.use_crx_entangle = use_crx_entangle
        self.use_adaptive_strength = use_adaptive_strength



class QuantumCBModule(nn.Module):
    def __init__(self, in_channels, n_qubits, device):
        super().__init__()
        self.C = in_channels
        self.n_qubits = n_qubits
        self.device = device

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.q_input_proj = nn.Linear(in_channels, n_qubits, bias=False).to(device)
        nn.init.xavier_uniform_(self.q_input_proj.weight)




def get_qlayer_output_dim(qlayer, input_dim, device):
    with torch.no_grad():
        return qlayer(torch.randn(1, input_dim).to(device)).size(-1)

class QuantumFeatHead(nn.Module):
    def __init__(self, classic_in_dim, q_input_dim, qlayer, device, use_EA=False, eps=0.1):
        super().__init__()
        self.device = device
        self.pre_linear = nn.Linear(classic_in_dim, q_input_dim).to(device)
        self.q_layer = qlayer.to(device)
        self.q_output_dim = get_qlayer_output_dim(qlayer, q_input_dim, device)
        self.proj = nn.Sequential(
            nn.Linear(self.q_output_dim,128), nn.ReLU(),
            nn.Linear(128,128)
        ).to(device)

    def forward(self, x):
        return F.normalize(self.proj(self.q_layer(self.pre_linear(x.to(self.device)))), dim=1)

def build_model(device, input_size=10, n_qubits=6):
    qcnn = MQCNN_Conv2d(1,5,2,input_size).to(device)
    cb = QuantumCBModule(5, n_qubits, device).to(device)
    pool = nn.MaxPool2d(2).to(device)

    with torch.no_grad():
        feat_dim = pool(cb(qcnn(torch.randn(1,1,input_size,input_size).to(device)))).flatten(1).size(1)

    perturb = PQCStylePerturb(feat_dim, n_qubits, device).to(device)

    sim14 = SimCirc14(layers=3, sweeps_per_layer=2)
    circ_func = sim14.get_quantum_circuit(n_qubits)
    if not hasattr(circ_func, "__name__"):
        circ_func.__name__ = "sim14_quantum_circuit"

    qnode = qml.QNode(circ_func, qml.device("default.qubit", wires=n_qubits), diff_method="best")
    weight_shapes = {"params": sim14.n_params}
    q_layer = TorchLayer(qnode, weight_shapes).to(device)
    head = QuantumFeatHead(feat_dim, n_qubits, q_layer, device).to(device)

    model = nn.Sequential(
        qcnn, cb, pool, nn.BatchNorm2d(5).to(device),
        perturb, nn.Flatten(), head
    ).to(device)

    return model

class LinearProbe(nn.Module):
    def __init__(self, encoder, feature_dim=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(feature_dim, num_classes)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x):
        with torch.no_grad():
            feat = self.encoder(x)
        return self.fc(feat)

def get_val_transform(input_size=10):
    return tv_transforms.Compose([
        tv_transforms.Resize((input_size, input_size)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((0.2860,), (0.3530,)),
    ])

def load_fmnist01_with_val(data_path="../FashionMNIST", total_per_cls=50, test_per_cls=30, val_ratio=0.2):
    transform = get_val_transform(10)
    full_train_ds = torchvision.datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    full_test_ds = torchvision.datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

    def get_cls_all_idx(ds, n_per_cls):
        idx0 = np.where(np.array(ds.targets)==0)[0][:n_per_cls]
        idx1 = np.where(np.array(ds.targets)==1)[0][:n_per_cls]
        return idx0, idx1

    idx0_all, idx1_all = get_cls_all_idx(full_train_ds, total_per_cls)

    def split_train_val(cls_idx, ratio):
        np.random.shuffle(cls_idx)
        split_point = int(len(cls_idx) * (1 - ratio))
        train_idx = cls_idx[:split_point]
        val_idx = cls_idx[split_point:]
        return train_idx, val_idx

    train0, val0 = split_train_val(idx0_all, val_ratio)
    train1, val1 = split_train_val(idx1_all, val_ratio)

    train_idx = np.concatenate([train0, train1])
    val_idx = np.concatenate([val0, val1])
    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_train_ds, val_idx)

    test0 = np.where(np.array(full_test_ds.targets)==0)[0][:test_per_cls]
    test1 = np.where(np.array(full_test_ds.targets)==1)[0][:test_per_cls]
    test_idx = np.concatenate([test0, test1])
    test_ds = Subset(full_test_ds, test_idx)

    return train_ds, val_ds, test_ds

def train_one_epoch(model, loader, opt, cri, device):
    model.train()
    model.encoder.eval()
    loss, corr, total = 0,0,0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        l = cri(out,y)
        l.backward()
        opt.step()
        loss += l.item() * x.size(0)
        corr += (out.argmax(1)==y).sum().item()
        total += x.size(0)
    return loss / total, corr / total

def validate(model, loader, cri, device):
    model.eval()
    loss, corr, total = 0,0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss += cri(out,y).item() * x.size(0)
            corr += (out.argmax(1)==y).sum().item()
            total += x.size(0)
    return loss / total, corr / total

def test(model, loader, cri, device):
    model.eval()
    loss, corr, total = 0,0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss += cri(out,y).item() * x.size(0)
            corr += (out.argmax(1)==y).sum().item()
            total += x.size(0)
    return loss / total, corr / total

if __name__ == "__main__":
    seed_everything(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_size = 10
    n_qubits = 6

    model = build_model(device, input_size, n_qubits)
    ckpt_path = "./Models_FUSION_QSEA_10/model_73.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅ 预训练编码器加载完成：{ckpt_path}")

    lp_model = LinearProbe(encoder=model, feature_dim=128, num_classes=2).to(device)

    train_ds, val_ds, test_ds = load_fmnist01_with_val(
        total_per_cls=100,
        test_per_cls=30,
        val_ratio=0.2
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lp_model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 80
    best_val_acc = 0.0

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []

    save_dir = "./linear_probe_final"
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "train_log-73-50perclass.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Epoch\tTrainLoss\tTrainAcc\tValLoss\tValAcc\tTestLoss\tTestAcc\tBestValAcc\n")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(lp_model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(lp_model, val_loader, criterion, device)
        te_loss, te_acc = test(lp_model, test_loader, criterion, device)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        print(f"Epoch {epoch:03d} | Train:{tr_acc:.4f} | Val:{val_acc:.4f} | Test:{te_acc:.4f} | BestVal:{best_val_acc:.4f}")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{tr_loss:.6f}\t{tr_acc:.4f}\t{val_loss:.6f}\t{val_acc:.4f}\t{te_loss:.6f}\t{te_acc:.4f}\t{best_val_acc:.4f}\n")
