import os
import torch
from training.data.text_vocab import VOCAB_SIZE
from bluecodec import LatentEncoder, LatentDecoder1D
from training.utils import LinearMelSpectrogram
from training.t2l.models.text_encoder import TextEncoder
from training.t2l.models.vf_estimator import VectorFieldEstimator
from training.t2l.models.reference_encoder import ReferenceEncoder
from training.dp.models.dp_network import DPNetwork
from training.t2l.cfg_utils import UncondParams

def build_models(ttl_cfg, ae_cfg_json, ae_sample_rate, device, dp_ckpt_path="checkpoints/duration_predictor/duration_predictor_final.pt"):
    te_cfg = ttl_cfg["text_encoder"]
    te_d_model = te_cfg["text_embedder"]["char_emb_dim"]
    te_convnext_layers = te_cfg["convnext"]["num_layers"]
    te_convnext_intermediate = te_cfg["convnext"]["intermediate_dim"]
    te_expansion_factor = te_convnext_intermediate // te_d_model
    te_attn_n_heads = te_cfg["attn_encoder"]["n_heads"]
    te_attn_n_layers = te_cfg["attn_encoder"]["n_layers"]
    te_attn_filter_channels = te_cfg["attn_encoder"]["filter_channels"]
    te_attn_p_dropout = te_cfg["attn_encoder"]["p_dropout"]

    se_cfg = ttl_cfg["style_encoder"]
    se_d_model = se_cfg["proj_in"]["odim"]
    se_hidden_dim = se_cfg["convnext"]["intermediate_dim"]
    se_num_blocks = se_cfg["convnext"]["num_layers"]
    se_n_style = se_cfg["style_token_layer"]["n_style"]
    se_n_heads = se_cfg["style_token_layer"]["n_heads"]
    se_prototype_dim = se_cfg["style_token_layer"].get("prototype_dim", 256)
    se_n_units = se_cfg["style_token_layer"].get("n_units", 256)
    se_style_value_dim = se_cfg["style_token_layer"].get("style_value_dim", 256)
    se_in_channels = se_cfg["proj_in"]["ldim"] * se_cfg["proj_in"]["chunk_compress_factor"]

    spte_cfg = ttl_cfg["speech_prompted_text_encoder"]
    spte_n_heads = spte_cfg["n_heads"]
    spte_text_dim = spte_cfg.get("text_dim", 256)
    spte_style_dim = spte_cfg.get("style_dim", 256)
    spte_n_units = spte_cfg.get("n_units", 256)
    spte_n_style = se_n_style

    um_cfg = ttl_cfg["uncond_masker"]
    uncond_init_std = um_cfg["std"]
    um_text_dim = um_cfg["text_dim"]
    um_n_style = um_cfg["n_style"]
    um_style_value_dim = um_cfg["style_value_dim"]

    vf_cfg = ttl_cfg["vector_field"]
    vf_in_channels = vf_cfg["proj_in"]["ldim"] * vf_cfg["proj_in"]["chunk_compress_factor"]
    vf_out_channels = vf_cfg["proj_out"]["ldim"] * vf_cfg["proj_out"]["chunk_compress_factor"]
    vf_hidden = vf_cfg["proj_in"]["odim"]
    vf_time_dim = vf_cfg["time_encoder"]["time_dim"]
    vf_time_hdim = vf_cfg["time_encoder"].get("hdim", 256)
    vf_n_blocks = vf_cfg["main_blocks"]["n_blocks"]
    vf_text_dim = vf_cfg["main_blocks"]["text_cond_layer"]["text_dim"]
    vf_text_n_heads = vf_cfg["main_blocks"]["text_cond_layer"]["n_heads"]
    vf_style_dim = vf_cfg["main_blocks"]["style_cond_layer"]["style_dim"]
    vf_rotary_scale = vf_cfg["main_blocks"]["text_cond_layer"]["rotary_scale"]
    vf_use_residual = vf_cfg["main_blocks"]["text_cond_layer"].get("use_residual", True)
    vf_rotary_base = vf_cfg["main_blocks"]["text_cond_layer"].get("rotary_base", 10000.0)

    te_ksz = te_cfg["convnext"].get("ksz", 5)
    te_dilation_lst = te_cfg["convnext"].get("dilation_lst", [1] * te_convnext_layers)

    se_ksz = se_cfg["convnext"].get("ksz", 5)
    se_dilation_lst = se_cfg["convnext"].get("dilation_lst", [1] * se_num_blocks)

    vf_main_blocks_cfg = vf_cfg.get("main_blocks", {})
    vf_last_convnext_cfg = vf_cfg.get("last_convnext", {})

    ae_enc_arch = ae_cfg_json['encoder']
    ae_spec_cfg = ae_enc_arch.get('spec_processor', {})
    hop_length = ae_spec_cfg.get('hop_length', 512)
    
    mel_spec = LinearMelSpectrogram(
        sample_rate=ae_spec_cfg.get('sample_rate', 44100),
        n_fft=ae_spec_cfg.get('n_fft', 2048),
        win_length=ae_spec_cfg.get('win_length', ae_spec_cfg.get('n_fft', 2048)),
        hop_length=ae_spec_cfg.get('hop_length', 512),
        n_mels=ae_spec_cfg.get('n_mels', 228),
    ).to(device)
    ae_encoder = LatentEncoder(cfg=ae_enc_arch).to(device)
    ae_decoder = LatentDecoder1D(cfg=ae_cfg_json['decoder']).to(device)

    text_encoder = TextEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=te_d_model,
        n_conv_layers=te_convnext_layers,
        n_attn_layers=te_attn_n_layers,
        expansion_factor=te_expansion_factor,
        p_dropout=te_attn_p_dropout,
        kernel_size=te_ksz,
        dilation_lst=te_dilation_lst,
        attn_n_heads=te_attn_n_heads,
        attn_filter_channels=te_attn_filter_channels,
        spte_n_heads=spte_n_heads,
        spte_text_dim=spte_text_dim,
        spte_style_dim=spte_style_dim,
        spte_n_units=spte_n_units,
        spte_n_style=spte_n_style,
    ).to(device)

    reference_encoder = ReferenceEncoder(
        in_channels=se_in_channels,
        d_model=se_d_model,
        hidden_dim=se_hidden_dim,
        num_blocks=se_num_blocks,
        num_tokens=se_n_style,
        num_heads=se_n_heads,
        kernel_size=se_ksz,
        dilation_lst=se_dilation_lst,
        prototype_dim=se_prototype_dim,
        n_units=se_n_units,
        style_value_dim=se_style_value_dim,
    ).to(device)

    vf_estimator = VectorFieldEstimator(
        in_channels=vf_in_channels,
        out_channels=vf_out_channels,
        hidden_channels=vf_hidden,
        text_dim=vf_text_dim,
        style_dim=vf_style_dim,
        num_style_tokens=se_n_style,
        num_superblocks=vf_n_blocks,
        time_embed_dim=vf_time_dim,
        rope_gamma=float(vf_rotary_scale),
        main_blocks_cfg=vf_main_blocks_cfg,
        last_convnext_cfg=vf_last_convnext_cfg,
        text_n_heads=vf_text_n_heads,
        time_hdim=vf_time_hdim,
        use_residual=vf_use_residual,
        rotary_base=vf_rotary_base,
    ).to(device)

    uncond_params = UncondParams(
        text_dim=um_text_dim,
        n_style=um_n_style,
        style_value_dim=um_style_value_dim,
        init_std=uncond_init_std,
    ).to(device)

    dp_model = None
    if os.path.exists(dp_ckpt_path):
        try:
            print(f"Loading Duration Predictor from {dp_ckpt_path}...")
            dp_model = DPNetwork(vocab_size=VOCAB_SIZE).to(device)
            dp_state = torch.load(dp_ckpt_path, map_location=device)
            model_state = dp_model.state_dict()
            filtered_state = {}
            for k, v in dp_state.items():
                if k in model_state:
                    if v.shape != model_state[k].shape:
                        if "char_embedder.weight" in k and v.shape[0] > model_state[k].shape[0]:
                            filtered_state[k] = v[:model_state[k].shape[0], :]
                        else:
                            print(f"Skipping DP {k} due to shape mismatch: {v.shape} vs {model_state[k].shape}")
                        continue
                    filtered_state[k] = v
            dp_model.load_state_dict(filtered_state, strict=False)
            dp_model.eval()
            dp_model.requires_grad_(False)
        except Exception as e:
            print(f"Failed to load DP: {e}")

    return text_encoder, reference_encoder, vf_estimator, uncond_params, dp_model, ae_encoder, ae_decoder, mel_spec, hop_length
