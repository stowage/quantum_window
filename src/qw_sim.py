import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import factorial

# ==========================================
# CONFIGURATION
# ==========================================
FOCK_DIM = 4
GRID_SIZE = 16
HISTORY_LEN = 5
PRED_SHIFT = 3
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Simulation running on: {DEVICE}")


# ==========================================
# MODULE 1: PHYSICS SIMULATOR
# ==========================================
class PhysicsSimulator:
    def __init__(self, size=GRID_SIZE, dim=FOCK_DIM):
        self.size = size
        self.dim = dim

    def _coherent_rho(self, alpha):
        if alpha == 0:
            rho = np.zeros((self.dim, self.dim), dtype=np.complex64)
            rho[0, 0] = 1.0
            return rho

        n = np.arange(self.dim)
        prefactor = np.exp(-0.5 * np.abs(alpha) ** 2)
        psi = prefactor * (np.power(alpha, n) / np.sqrt(factorial(n)))
        return np.outer(psi, np.conj(psi))

    def generate_sequence(self, steps=100):
        data = np.zeros((steps, 2, self.dim, self.dim, self.size, self.size), dtype=np.float32)
        center = self.size // 2
        radius = self.size // 4

        print(f"Generating physics simulation ({steps} frames)...")

        for t in range(steps):
            angle = 0.2 * t
            x_pos = center + radius * np.cos(angle)
            y_pos = center + radius * np.sin(angle)

            for y in range(self.size):
                for x in range(self.size):
                    dist = (x - x_pos) ** 2 + (y - y_pos) ** 2
                    amplitude = 2.0 * np.exp(-dist / 5.0)
                    phase = np.exp(1j * 0.5 * t)
                    alpha = amplitude * phase

                    rho = self._coherent_rho(alpha)

                    # Noise
                    noise_lvl = 0.05
                    noise = (np.random.randn(self.dim, self.dim) +
                             1j * np.random.randn(self.dim, self.dim)) * noise_lvl
                    # Ensure Hermitian symmetry for input data
                    noise = (noise + noise.conj().T) / 2
                    mask = 1 - np.eye(self.dim)
                    rho += noise * mask

                    data[t, 0, :, :, y, x] = rho.real
                    data[t, 1, :, :, y, x] = rho.imag

        return torch.tensor(data, device=DEVICE)

    def readout_diagonal(self, q_tensor):
        rho_real = q_tensor[0]
        intensity = torch.zeros((self.size, self.size), device=DEVICE)
        for n in range(self.dim):
            intensity += n * rho_real[n, n, :, :]
        return intensity.cpu().numpy()


# ==========================================
# MODULE 2: LOSS FUNCTION
# ==========================================
class QuantumFidelityLoss(nn.Module):
    """
    Fidelity Loss.
    """

    def __init__(self, fock_dim, epsilon=1e-6):
        super(QuantumFidelityLoss, self).__init__()
        self.fock_dim = fock_dim
        self.epsilon = epsilon

    def forward(self, pred_tensor, target_tensor):
        # 1. Convert to Complex: [Batch, H, W, D, D]
        rho_pred = self._to_complex(pred_tensor)
        rho_target = self._to_complex(target_tensor)

        # 2. Compute Overlap: Tr(A @ B)
        product = torch.matmul(rho_pred, rho_target)
        overlap = torch.diagonal(product, dim1=-2, dim2=-1).sum(-1).real

        # 3. Compute Generalized Purity (Norm Squared): Tr(A^dagger @ A)
        term_p = torch.matmul(rho_pred.conj().transpose(-2, -1), rho_pred)
        purity_p = torch.diagonal(term_p, dim1=-2, dim2=-1).sum(-1).real

        term_t = torch.matmul(rho_target.conj().transpose(-2, -1), rho_target)
        purity_t = torch.diagonal(term_t, dim1=-2, dim2=-1).sum(-1).real

        # 4. Fidelity calculation
        denominator = torch.sqrt(purity_p * purity_t + self.epsilon)
        fidelity = overlap / denominator

        # 5. Loss
        return 1.0 - torch.mean(fidelity)

    def _to_complex(self, t):
        # Combine Real/Imag
        comp = torch.complex(t[:, 0], t[:, 1])  # [B, D, D, H, W]
        # Permute to [B, H, W, D, D]
        return comp.permute(0, 3, 4, 1, 2)


# ==========================================
# MODULE 3: NEURAL NETWORK
# ==========================================
class QuantumPredictNet(nn.Module):
    def __init__(self, fock_dim, hidden_dim=64):
        super(QuantumPredictNet, self).__init__()
        self.fock_dim = fock_dim
        self.in_channels = 2 * fock_dim ** 2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels * HISTORY_LEN, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, 1, 1),
            nn.ReLU()
        )
        self.time_emb = nn.Sequential(nn.Linear(1, hidden_dim * 2), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, self.in_channels, 1)
        )

    def forward(self, x, dt):
        b, t, c, d1, d2, h, w = x.shape
        x_flat = x.view(b, -1, h, w)

        # 1. Encode
        feat = self.encoder(x_flat)
        t_vec = self.time_emb(dt).unsqueeze(-1).unsqueeze(-1)

        # 2. Decode Raw Output (Unconstrained numbers)
        raw_out = self.decoder(feat * (1.0 + t_vec))  # [B, 2*D*D, H, W]

        # 3. PHYSICS ENFORCEMENT LAYER (The Fix)
        # Convert flat raw output to Complex Matrix M
        M = self._to_complex(raw_out)  # [B, H, W, D, D]

        M_dagger = M.conj().transpose(-2, -1)
        rho_unscaled = torch.matmul(M_dagger, M)

        # Normalize Trace to 1 (Conservation of Probability)
        # Trace is sum of diagonal elements
        trace = torch.diagonal(rho_unscaled, dim1=-2, dim2=-1).sum(-1)
        # Avoid division by zero
        trace = trace.unsqueeze(-1).unsqueeze(-1) + 1e-6

        rho = rho_unscaled / trace

        # 4. Flatten back to [B, 2, D, D, H, W] for the rest of the code
        return self._to_flat(rho)

    def _to_complex(self, t):
        # [B, 2*D*D, H, W] -> [B, H, W, D, D] Complex
        b, c, h, w = t.shape
        t = t.permute(0, 2, 3, 1).contiguous().view(b, h, w, 2, self.fock_dim, self.fock_dim)
        return torch.complex(t[..., 0, :, :], t[..., 1, :, :])

    def _to_flat(self, rho):
        # [B, H, W, D, D] Complex -> [B, 2, D, D, H, W] Real
        b, h, w, d, _ = rho.shape
        rho = rho.permute(0, 3, 4, 1, 2)  # [B, D, D, H, W]
        res = torch.stack([rho.real, rho.imag], dim=1)  # [B, 2, D, D, H, W]
        return res


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # SETUP
    sim = PhysicsSimulator()
    model = QuantumPredictNet(FOCK_DIM).to(DEVICE)
    loss_fn = QuantumFidelityLoss(FOCK_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # DATA
    raw_data = sim.generate_sequence(steps=200)
    inputs, targets, shifts = [], [], []
    valid_len = len(raw_data) - HISTORY_LEN - PRED_SHIFT

    for i in range(valid_len):
        inputs.append(raw_data[i: i + HISTORY_LEN])
        targets.append(raw_data[i + HISTORY_LEN + PRED_SHIFT])
        shifts.append(float(PRED_SHIFT))

    inputs = torch.stack(inputs).to(DEVICE)
    targets = torch.stack(targets).to(DEVICE)
    shifts = torch.tensor(shifts).unsqueeze(1).to(DEVICE)

    # TRAIN
    print(f"\n--- Training on {len(inputs)} samples ---")
    model.train()
    loss_history = []

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(inputs, shifts)
        loss = loss_fn(pred, targets)
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        fidelity = (1.0 - loss.item()) * 100
        loss_history.append(loss.item())

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Fidelity: {fidelity:.2f}% | Loss: {loss.item():.4f}")

    # TEST
    print("\n--- Running Prediction Test ---")
    model.eval()
    test_data = sim.generate_sequence(steps=30)

    idx = 10
    hist = test_data[idx: idx + HISTORY_LEN].unsqueeze(0)
    dt = torch.tensor([[float(PRED_SHIFT)]]).to(DEVICE)

    with torch.no_grad():
        pred_rho = model(hist, dt)

    true_future = test_data[idx + HISTORY_LEN + PRED_SHIFT]
    current_frame = test_data[idx + HISTORY_LEN - 1]

    img_now = sim.readout_diagonal(current_frame)
    img_pred = sim.readout_diagonal(pred_rho[0])
    img_true = sim.readout_diagonal(true_future)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1);
    plt.title("Loss Curve");
    plt.plot(loss_history)
    plt.subplot(1, 4, 2);
    plt.title("Current (t)");
    plt.imshow(img_now, cmap='magma', vmin=0)
    plt.subplot(1, 4, 3);
    plt.title(f"AI Predict (t+{PRED_SHIFT})");
    plt.imshow(img_pred, cmap='magma', vmin=0)
    plt.subplot(1, 4, 4);
    plt.title(f"Truth (t+{PRED_SHIFT})");
    plt.imshow(img_true, cmap='magma', vmin=0)

    plt.tight_layout()
    plt.show()
