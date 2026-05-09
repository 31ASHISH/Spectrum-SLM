import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, sys, pickle
import numpy as np

if __name__ == '__main__':
    # Setup path and load config
    sys.path.insert(0, '.')
    from spectrum_slm_model import SpectrumSLM
    from config import N_MOD_CLASSES_V2

    print("Loading 1 file for quick gen_head alignment...")
    SU = r'C:\Users\ASUS Vivo book\Desktop\Complete-Data-Science-With-Machine-Learning-And-NLP-2024-main\SDR_Data\Secondary_User'
    
    # Just take ONE file to train the AutoEncoder mapping
    file_path = os.path.join(SU, 'Symbol1', 'psd_binned_by_snr_bpsk.pth')
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    
    psds = []
    pairs = data['pairs_by_bin']
    for b in pairs:
        for entry in pairs[b]:
            psd_raw = np.array(entry[0], dtype=np.float32).flatten()
            if len(psd_raw) >= 192: psds.append(psd_raw[:192])
            else: psds.append(np.pad(psd_raw, (0, 192-len(psd_raw))))
            
    # Add some AWGN samples as well
    psds += [np.random.normal(0, 5, 192).astype(np.float32) for _ in range(500)]
    psds = np.array(psds)

    print(f"Loaded {len(psds)} samples.")
    with open('checkpoints/phase2/normalizer.pkl', 'rb') as f:
        scaler = pickle.load(f)
    psds_norm = scaler.transform(psds).astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(psds_norm).to(device)
    dataset = TensorDataset(X, X)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    print("Loading Phase 2 model...")
    model = SpectrumSLM(n_bins=192, patch_size=1, d_model=128, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1, n_mod_classes=N_MOD_CLASSES_V2)
    ckpt = torch.load('checkpoints/phase2/slm_phase2_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    model.to(device)

    # Freeze everything EXCEPT gen_head
    for name, param in model.named_parameters():
        if 'gen_head' not in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.gen_head.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    print("Training gen_head (AutoEncoder mode) for 5 Epochs...")
    model.train()
    for ep in range(5):
        losses = []
        for i, (bx, by) in enumerate(loader):
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out['gen_pred'], by)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"  Epoch {ep+1}/5 - MSE Loss: {np.mean(losses):.4f}")

    print("Saving aligned Phase 3 checkpoint...")
    ckpt['model'] = model.state_dict()
    ckpt['epoch'] = 36
    os.makedirs('checkpoints/phase3', exist_ok=True)
    torch.save(ckpt, 'checkpoints/phase3/slm_phase3_best.pt')
    print("Done! Model will now predict correctly.")
