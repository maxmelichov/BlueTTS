import os
import torch
from training.utils import LinearMelSpectrogram
from bluecodec import LatentEncoder
from training.dp.models.dp_network import DPNetwork
from training.data.text_vocab import VOCAB_SIZE

def build_dp_models(full_cfg, style_tokens, style_dim, sentence_encoder_cfg, style_encoder_cfg, predictor_cfg, ae_checkpoint, device):
    ae_enc_arch = full_cfg['ae']['encoder']
    ae_spec_cfg = ae_enc_arch.get('spec_processor', {})
    mel_spec = LinearMelSpectrogram(
        sample_rate=ae_spec_cfg.get('sample_rate', 44100),
        n_fft=ae_spec_cfg.get('n_fft', 2048),
        win_length=ae_spec_cfg.get('win_length', ae_spec_cfg.get('n_fft', 2048)),
        hop_length=ae_spec_cfg.get('hop_length', 512),
        n_mels=ae_spec_cfg.get('n_mels', 228),
    ).to(device)
    ae_encoder = LatentEncoder(cfg=ae_enc_arch).to(device)
    
    if os.path.exists(ae_checkpoint):
        print(f"Loading AE checkpoint from {ae_checkpoint}")
        if ae_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(ae_checkpoint)
        else:
            ckpt = torch.load(ae_checkpoint, map_location="cpu")
        
        if "encoder" in ckpt:
            ae_encoder.load_state_dict(ckpt["encoder"])
        elif "state_dict" in ckpt:
            ae_encoder.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            if any(k.startswith("encoder.") for k in ckpt.keys()):
                enc_dict = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                ae_encoder.load_state_dict(enc_dict)
            else:
                try:
                    ae_encoder.load_state_dict(ckpt)
                except Exception as e:
                    print(f"Could not load AE encoder: {e}")
    else:
        print("Warning: AE checkpoint not found. Latent quality may be poor.")
        
    ae_encoder.eval()
    ae_encoder.requires_grad_(False)
    mel_spec.eval()
    
    model = DPNetwork(
        vocab_size=VOCAB_SIZE,
        style_tokens=style_tokens,
        style_dim=style_dim,
        sentence_encoder_cfg=sentence_encoder_cfg,
        style_encoder_cfg=style_encoder_cfg,
        predictor_cfg=predictor_cfg,
    ).to(device)
    
    return mel_spec, ae_encoder, model