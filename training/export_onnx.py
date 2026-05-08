"""
training/export_onnx.py
=======================
ONNX export + verification for Spectrum-SLM.

CLI:
    python training/export_onnx.py
    python training/export_onnx.py --ckpt checkpoints/phase2/slm_phase2_best.pt

Authors : Anjani, Ashish Joshi, Mayank
Dated   : May 2026
"""

import os, sys, argparse, timeit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from config import (
    N_BINS, D_MODEL, N_HEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
    N_MOD_CLASSES_V2, CKPT_PHASE2, CKPT_PHASE2_BEST,
)
from spectrum_slm_model import SpectrumSLM


def export_onnx(
    ckpt_path: str,
    save_path: str = 'spectrum_slm.onnx',
    opset:     int = 17,
) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  [WARN] Run: pip install onnx onnxruntime")
        return

    model = SpectrumSLM(n_bins=N_BINS, patch_size=1, d_model=D_MODEL,
                        nhead=N_HEAD, num_layers=NUM_LAYERS,
                        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
                        n_mod_classes=N_MOD_CLASSES_V2)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt.get('model', ckpt))
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  [WARN] No checkpoint found at {ckpt_path} — exporting untrained model")

    model.eval()
    dummy = torch.randn(1, N_BINS)

    print(f"\nExporting ONNX → {save_path}")
    torch.onnx.export(
        model, (dummy,), save_path,
        opset_version = opset,
        input_names   = ['psd_input'],
        output_names  = ['pu_logits', 'mod_logits', 'snr_pred', 'gen_pred', 'cls_feat'],
        dynamic_axes  = {
            'psd_input' : {0: 'batch'},
            'pu_logits' : {0: 'batch'},
            'mod_logits': {0: 'batch'},
            'snr_pred'  : {0: 'batch'},
            'gen_pred'  : {0: 'batch'},
            'cls_feat'  : {0: 'batch'},
        },
    )

    # Verify
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model verified")

    # Run test
    sess = ort.InferenceSession(save_path,
           providers=['CPUExecutionProvider'])
    outs = sess.run(None, {'psd_input': dummy.numpy()})
    print(f"  ✓ Inference OK — shapes: {[o.shape for o in outs]}")

    # Latency
    n_iter = 200
    t = timeit.timeit(
        lambda: sess.run(None, {'psd_input': dummy.numpy()}),
        number=n_iter)
    print(f"  ✓ Latency: {1000*t/n_iter:.2f} ms / sample")
    print(f"  Saved → {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spectrum-SLM ONNX Export')
    parser.add_argument('--ckpt', default=os.path.join(CKPT_PHASE2, CKPT_PHASE2_BEST))
    parser.add_argument('--out',  default='spectrum_slm.onnx')
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()
    export_onnx(args.ckpt, args.out, args.opset)
