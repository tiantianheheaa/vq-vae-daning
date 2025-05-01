# from: https://adaning.github.io/posts/62916.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, patch_size):
        super().__init__()
        self.cnn = nn.Conv2d(input_size, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.linear(x.permute(0, 2, 3, 1))  # [b, c, h, w] -> [b, h, w, c]
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, patch_size):
        super().__init__()
        # Decoder使用的是反卷积
        self.cnn = nn.ConvTranspose2d(input_size, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.linear(x.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2)


class VQVAE(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, codebook_size, patch_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.patch_size = patch_size

        self.encoder = Encoder(input_size, hidden_size, hidden_size, patch_size=patch_size)
        self.decoder = Decoder(input_size, output_size, hidden_size, patch_size=patch_size)
        self.codebook = nn.Embedding(codebook_size, hidden_size)
        # codebook初始化
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def get_codebook_embeddings(self):
        return self.codebook.weight

    def forward(self, x):
        batch_size = x.size(0)
        z_e = self.encoder(x)
        sqrt_patches = z_e.size(1)

        # VQ
        z_e = z_e.view(-1, self.hidden_size)
        embeddings = self.get_codebook_embeddings()
        nearest = torch.argmin(torch.cdist(z_e, embeddings), dim=1)  # index
        z_q = self.codebook(nearest)  # 从embedding中根据index查到emb

        # STE: straight through estimator 直通估计
        decoder_input = z_e + (z_q - z_e).detach()

        decoder_input = decoder_input.view(batch_size, sqrt_patches, sqrt_patches, self.hidden_size)
        decoder_input = decoder_input.permute(0, 3, 1, 2)
        x_hat = F.sigmoid(self.decoder(decoder_input))

        return x_hat, z_e, z_q


epochs = 200
batch_size = 128
hidden_size = 256
codebook_size = 128
input_size = 1
patch_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "gpu")
lr = 1e-4
beta = 0.25

# 数据
transform = transforms.Compose([transforms.ToTensor])
data_train = MNIST("MNIST_DATA/", train=True, download=True, transform=transform)
data_valid = MNIST("MNIST_DATA/", train=False, download=True, transform=transform)

# num_workers用于DataLoader类，用于控制数据加载的子进程数（工作线程）。num_workers=0表示在主进程中加载，无并行。
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, num_workers=0)

# 模型
model = VQVAE(input_size=input_size, output_size=input_size, hidden_size=hidden_size, codebook_size=codebook_size, patch_size=patch_size)

# model.to(device)的作用
# 1.设备一致性：pytorch要求模型和输入数据必须在同一设备上（cpu或gpu）。例如 input_data = torch.randn(1,10).to(device)
# 2.加速训练：GPU通常比CPU快很多。
# 3.多GPU训练：cuda:0，cuda:1等，例如使用nn.DataParallel 或 nn.DistributedDataParallel时。
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = nn.MSELoss()

train_losses = []
valid_losses = []

best_loss = 1e9
best_epoch = 0

# 训练
for epoch in range(epochs):
    print(f"Epoch {epoch}")
    model.train()
    train_loss = 0.0

    for idx, (x,_) in enumerate(train_loader):
        # 数据和模型都要to(device)
        x = x.to(device)
        current_batch = x.size(0)
        x_hat, z_e, z_q = model(x)
        # 重建loss：整个任务自回归的loss
        recon_loss = criterion(x_hat, x)
        vq_loss = criterion(z_q, z_e.detach())
        commit_loss = criterion(z_e, z_q.detach())
        # loss是三部分相加
        loss = recon_loss + vq_loss + beta*commit_loss
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(f"training loss: {loss: .3f}, recon loss: {recon_loss.item(): .3f} in step {idx}")

    train_losses.append(train_loss / idx)

    # valid模式
    valid_loss, valid_recon, valid_vq = 0.0, 0.0, 0.0
    model.eval()

    with torch.no_grad():
        for idx, (x, _) in enumerate(test_loader):
            x = x.to(device)
            current_batch = x.size(0)
            x_hat, z_e, z_q = model(x)
            recon_loss = criterion(x_hat, x)
            vq_loss = criterion(z_q, z_e.detach())
            commit_loss = criterion(z_e, z_q.detach())
            loss = recon_loss + vq_loss + beta*commit_loss
            valid_loss += loss.item()
            valid_recon += recon_loss.item()
            valid_vq += vq_loss.item()

        valid_losses.append(valid_loss / idx)
        print(f"valid loss {valid_loss: .3f}, recon loss {valid_recon: .3f} in epoch{epoch}")

        # valid loss变小，就会保存当前最优模型
        if valid_recon < best_loss:
            best_loss = valid_recon
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model_mnist")
            print("model saved.")


# 训练loss可视化
plt.plot(train_losses, label="Train")
plt.plot(valid_losses, label="Valid")
plt.legend()
plt.title("Loss Curve")
plt.show()

# 图像可视化
imgs = []
batch_size = 10
test_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=True, num_workers=0)

model.load_state_dict(torch.load("./best_model_mnist"))
model.to(device)
with torch.no_grad():
    model.eval()
    for idx, (x, _) in enumerate(test_loader)
        if idx == 10:
            break
        x = x.to(device)
        x_org = x
        x_hat, z_e, z_q = model(x)
        imgs.append(x_org)
        imgs.append(x_hat)

res = torchvision.utils.make_grid(torch.cat(imgs, dim=0), nrow=batch_size)
img = torchvision.transforms.ToPILImage()(res)
img

with torch.no_grad():
    model.eval()
    decoder_input = model.get_codebook_embeddings().data[:, :, None, None]
    x_hat = torch.nn.functional.sigmoid(model.decoder(decoder_input))
    res = torchvision.utils.make_grid(x_hat, nrow=batch_size)
    img = torchvision.transforms.ToPILImage()(res)

img


