import shutil, os, torch

p2     = 'checkpoints/phase2/slm_phase2_best.pt'
p3_dir = 'checkpoints/phase3'
p3     = os.path.join(p3_dir, 'slm_phase3_best.pt')

os.makedirs(p3_dir, exist_ok=True)
shutil.copy2(p2, p3)
print(f'Copied: {p2} -> {p3}')

ck    = torch.load(p3, map_location='cpu', weights_only=False)
state = ck.get('model', ck)
proj  = state['tokenizer.projection.weight']
epoch = ck.get('epoch', '?')
print(f'Phase 3 arch : {proj.shape}')
print(f'Epoch        : {epoch}')
if proj.shape == (128, 1):
    print('ARCH: NEW (192-bin) - CORRECT! Phase 3 is ready.')
else:
    print('ARCH: OLD - something went wrong.')
