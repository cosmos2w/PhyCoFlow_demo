import h5py, sys, torch, os
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from neuralop.models import FNO
from torchdiffeq import odeint
import numpy as np
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# ========================== 1. DATA LOADING & RESHAPING ==========================
def load_and_reshape_h5(h5_path="Dataset/Merged_CH4COTU1P.h5", grid_h=100, grid_w=403):
    with h5py.File(h5_path, 'r') as f:
        fields = f['fields'][:]          # (1, 10000, 40300, 1, 1, 5)
        coords = f['coordinates'][:]     # (40300, 1, 1, 3)  -- we ignore for grid assumption
        
        fields = fields.squeeze()        # (10000, 40300, 5)
        # Reshape to grid: row-major ordering (we can verify with coords if needed)
        fields = fields.reshape(10000, grid_h, grid_w, 5)  # (Nt, H, W, C)
    
    # Permute to standard FNO shape: (B, C, H, W), we treat each time step as a sample
    data = torch.from_numpy(fields).permute(0, 3, 1, 2).float()  # (10000, 5, 100, 403)
    
    # Normalize per channel
    mean = data.mean(dim=(0, 2, 3), keepdim=True)
    std  = data.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    data = (data - mean) / std
    
    print(f"Loaded dataset: {data.shape} → (N_samples, C=5, H=100, W=403)")
    return data, mean, std

# ========================== 2. FNO MODEL (original FFM style) ==========================
class FFM_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fno = FNO(
            in_channels=5 + 1,   # 5 fields + time
            out_channels=5,
            n_modes=(32, 8),
            hidden_channels=64,
            n_layers=4
        )
        self.time_embed = lambda t: t.view(-1, 1, 1, 1).expand(-1, -1, 100, 403)
    
    def forward(self, t, x):
        t_emb = self.time_embed(t)
        inp = torch.cat([x, t_emb], dim=1)
        return self.fno(inp)

# ========================== 3. TRAINING LOOP (original FFM OT path) ==========================
def train():

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    data, mean, std = load_and_reshape_h5()
    
    # Use first 9000 steps for train, last 1000 for val
    train_data = data[:9000]
    dataset = TensorDataset(train_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    model = FFM_Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    print("=== Starting FFM Training (original paper logic) ===")
    for epoch in range(100):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for (batch,) in pbar:
            x1 = batch.to(device) # (B, 5, 100, 403)
            t = torch.rand(x1.shape[0], device=device)
            
            # OT path in original FFM
            z = torch.randn_like(x1) * 0.01  # small Matérn-like noise (can replace with proper GP)
            xt = (1 - t).view(-1, 1, 1, 1) * z + t.view(-1, 1, 1, 1) * x1
            
            v_pred = model(t, xt)
            target_v = x1 - z  # exact conditional velocity
            
            loss = F.mse_loss(v_pred, target_v)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())
        
        # Save checkpoint every 50 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./Save_TrainedModel/ffm_checkpoint.pt")
            print(f'loss is {loss.item()}, saving checkpoint at epoch {epoch}')
    
    print("Training finished. Model saved.")
    return model, mean, std

# ========================== 4. GUIDED SAMPLING (arbitrary sparse measurements on ONE field) ==========================
@torch.no_grad()
def sample_guided(model, mean, std, n_samples=4, n_steps=8, obs_field_idx=2, obs_ratio=0.15):
    """obs_field_idx=2 → Temperature (T). Change to 0=CH4, 3=U_1, etc."""
    device = next(model.parameters()).device
    model.eval()
    
    # Start from noise
    x = torch.randn(n_samples, 5, 100, 403, device=device) * 0.01
    
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((n_samples,), step * dt, device=device)
        
        v = model(t, x)
        
        # === Sparse observation guidance on ONE field only ===
        mask = torch.rand(100, 403, device=device) < obs_ratio
        obs_noise = torch.randn_like(x[:, obs_field_idx]) * 0.05
        y_obs = (x[:, obs_field_idx] + obs_noise) * mask.unsqueeze(0)
        
        data_grad = torch.zeros_like(x)
        data_grad[:, obs_field_idx] = (x[:, obs_field_idx] - y_obs) * mask.unsqueeze(0)
        
        # Euler step with guidance
        x = x + dt * (v - 2.0 * data_grad)  # λ_data = 2.0 (tune this)
    
    # Denormalize
    x = x * std.to(device) + mean.to(device)
    return x

# ========================== RUN ==========================
if __name__ == "__main__":
    model, mean, std = train()
    
    # Example: generate 4 samples guided by 5% sparse temperature sensors
    generated = sample_guided(model, mean, std, n_samples=4, obs_field_idx=2, obs_ratio=0.05)
    torch.save(generated, "guided_samples.pt")
    print("Saved guided samples! Shape:", generated.shape)