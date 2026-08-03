import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"디바이스: {device}  |  PyTorch: {torch.__version__}")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 1. 운영체제별 폰트 설정
if platform.system() == "Windows":
    # 윈도우 맑은 고딕
    font_family = "Malgun Gothic"
elif platform.system() == "Darwin":
    # 맥 애플고딕
    font_family = "AppleGothic"
else:
    # 리눅스/코랩 나눔고딕
    font_family = "NanumGothic"

# 2. 폰트 설정
plt.rc("font", family=font_family)
# 3. 마이너스 부호 깨짐 해결
plt.rcParams["axes.unicode_minus"] = False


class VAE(nn.Module):
    """Convolutional VAE — 월드 모델의 Vision 모듈이다."""

    def __init__(self, in_ch=3, z_dim=32, img_size=64):
        super().__init__()
        self.z_dim, self.img_size = z_dim, img_size

        # 인코더: (B, C, H, W) -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1),
            nn.ReLU(),  # H/2
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),  # H/4
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),  # H/8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),  # H/16
            nn.Flatten(),
        )
        feat = 256 * (img_size // 16) ** 2
        self.fc_mu = nn.Linear(feat, z_dim)
        self.fc_logvar = nn.Linear(feat, z_dim)

        # 디코더: z -> (B, C, H, W)
        self.fc_dec = nn.Linear(z_dim, feat)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, img_size // 16, img_size // 16)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(self.fc_dec(z))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        return self.decode(z), mu, lv, z

    @staticmethod
    def loss_fn(x, xr, mu, lv, beta=1.0):
        recon = F.binary_cross_entropy(xr, x, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + lv - mu**2 - lv.exp()) / x.size(0)
        return recon + beta * kl, recon, kl


vae = VAE(3, 8, 64).to(device)
print(f"VAE 파라미터: {sum(p.numel() for p in vae.parameters()):,}")


# 합성 데이터 생성: 다양한 색/위치/크기의 원
def make_data(n=2000, sz=64):
    imgs = np.zeros((n, 3, sz, sz), dtype=np.float32)
    labs = np.zeros((n, 6), dtype=np.float32)
    yy, xx = np.mgrid[0:sz, 0:sz]
    for i in range(n):
        cx, cy = np.random.randint(14, sz - 14, 2)
        r = np.random.randint(5, 14)
        c = np.random.rand(3)
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r**2
        for ch in range(3):
            imgs[i, ch] = mask * c[ch]
        labs[i] = [cx / sz, cy / sz, r / 14, *c]
    return imgs, labs


print("합성 이미지 생성 중...")
X_np, Y_np = make_data(2000, 64)
X_train = torch.tensor(X_np, device=device)

fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for i in range(12):
    axes[i // 6, i % 6].imshow(X_np[i].transpose(1, 2, 0))
    axes[i // 6, i % 6].axis("off")

plt.suptitle("합성 학습 데이터", fontweight="bold")
plt.tight_layout()
plt.show()


# VAE 학습
def train_vae(model, data, epochs=50, bs=64, lr=1e-3, beta=0.5):
    opt = optim.Adam(model.parameters(), lr=lr)
    n = data.size(0)
    hist = {"total": [], "recon": [], "kl": []}
    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n, device=data.device)
        tot, rec, kld, nb = 0, 0, 0, 0
        for s in range(0, n, bs):
            b = data[idx[s : s + bs]]
            xr, mu, lv, _ = model(b)
            loss, rl, kl = model.loss_fn(b, xr, mu, lv, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            rec += rl.item()
            kld += kl.item()
            nb += 1
        hist["total"].append(tot / nb)
        hist["recon"].append(rec / nb)
        hist["kl"].append(kld / nb)
        if ep % 10 == 0:
            print(
                f"  Epoch {ep:3d}  Total={tot/nb:.1f}  Recon={rec/nb:.1f}  KL={kld/nb:.2f}"
            )
    return hist


# VAE 학습
print("VAE 학습 시작 (beta=0.5)...")
hist = train_vae(vae, X_train, 50, beta=0.5)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, k, c, t in zip(
    axes,
    ["total", "recon", "kl"],
    ["#2C3E50", "#E74C3C", "#27AE60"],
    ["Total Loss", "재구성 손실", "KL 발산"],
):
    ax.plot(hist[k], color=c, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_title(t, fontweight="bold")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 재구성 결과 + 잠재 공간 시각화
vae.eval()
with torch.no_grad():
    idx = np.random.choice(len(X_train), 6, replace=False)
    samp = X_train[idx]
    rec, _, _, _ = vae(samp)

fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for i in range(6):
    axes[0, i].imshow(samp[i].cpu().numpy().transpose(1, 2, 0))
    axes[0, i].set_title("원본", fontsize=9)
    axes[0, i].axis("off")
    axes[1, i].imshow(rec[i].cpu().numpy().transpose(1, 2, 0))
    axes[1, i].set_title("재구성", fontsize=9)
    axes[1, i].axis("off")
plt.suptitle("VAE 재구성 결과", fontweight="bold")
plt.tight_layout()
plt.show()

# 잠재 공간 2D 투영
with torch.no_grad():
    _, mu_all, _, _ = vae(X_train[:800])
    z_np = mu_all.cpu().numpy()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, ci, cn in zip(axes, [3, 4, 5], ["Red", "Green", "Blue"]):
    sc = ax.scatter(
        z_np[:, 0], z_np[:, 1], c=Y_np[:800, ci], cmap="viridis", alpha=0.5, s=12
    )
    plt.colorbar(sc, ax=ax, label=cn)
    ax.set_xlabel("z_1")
    ax.set_ylabel("z_2")
    ax.set_title(f"{cn} 채널에 따른 잠재 분포")
    ax.grid(alpha=0.2)
plt.suptitle("VAE 잠재 공간 (처음 2차원)", fontweight="bold")
plt.tight_layout()
plt.show()

# 이론 공부 제대로 하기 전에는 더 이상은 안되겠다...
