"""FlowTTS-GRPO post-training of the T2L vector field (arXiv 2606.23190).

Per iteration: `prompts_per_iter` (text, reference clip) prompts, each rolled out as `prompt_repeat` groups, 30% of
them hard cases (Sec 2.5, Table 1); G latents per group from the sampler with an SDE window (flow_grpo.py; without CFG
unless --rollout_cfg); decode, score (rewards.py), advantages, then `updates_per_iter` clipped-ratio updates with a KL
to the frozen starting VF.

Only the vector field trains, so voices exported from the base checkpoint stay valid. With the projected style key
the VF's `tile` is also the text encoder's key, so it is saved on both sides.
"""

import copy
import json
import os
import queue
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from training.data.text_vocab import text_to_indices, wrap_lang_tags
from training.rl import flow_grpo as fg
from training.t2l.builders import build_models
from training.t2l.data_module import Text2LatentDataset
from training.t2l.sampling import build_reference_only, get_latent_mask
from training.utils import compress_latents, decompress_latents

_PUNCT = r"[.,!?;:\"]"


def _split_punct(w):
    m = re.match(r"^(.*?)(" + _PUNCT + r"*)$", w)
    return m.group(1), m.group(2)


def hard_variant(ipa, text, kind, rng):
    """Sec 2.5 / Table 1 hard case of one prompt, at the same word positions in the IPA and (when given) its text:
    lwr one word x3-5; smr 2-3 words x2-3 (punctuation on the last copy); gsr the sentence twice after a comma.
    lwr / smr fall back to gsr when the positions cannot be matched (the IPA and the text differ in word count, or
    the text has digits, hyphens, dashes or symbols) or no word has 2+ letters. Returns (kind, ipa, text)."""
    iw = ipa.split()
    tw = text.split() if text else None
    if kind in ("lwr", "smr") and text and (len(tw) != len(iw) or re.search(r"[0-9\-\u2013\u2014/&%$]", text)):
        kind = "gsr"
    elig = [j for j, w in enumerate(iw) if len(_split_punct(w)[0]) >= 2]
    if not elig:
        kind = "gsr"
    if kind == "gsr":
        def twice(x):
            x = x.strip()
            return f"{re.sub(_PUNCT + '+$', '', x)}, {x}" + ("" if re.search(_PUNCT + "$", x) else ".")
        return kind, twice(ipa), twice(text) if text else text
    if kind == "lwr":
        picks = {rng.choice(elig): rng.randint(3, 5)}
    else:
        picks = {j: rng.randint(2, 3) for j in rng.sample(elig, min(len(elig), rng.choice([2, 3])))}

    def apply(ws):
        ws = list(ws)
        for j in sorted(picks, reverse=True):
            w, p = _split_punct(ws[j])
            ws[j:j + 1] = [w] * (picks[j] - 1) + [w + p]
        return " ".join(ws)
    return kind, apply(iw), apply(tw) if text else text


def _load_ae(ae_encoder, ae_decoder, path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        ck = load_file(path, device="cpu")
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    for mod, name in ((ae_encoder, "encoder"), (ae_decoder, "decoder")):
        sd = ck[name] if name in ck else {k[len(name) + 1:]: v for k, v in ck.items() if k.startswith(name + ".")}
        mod.load_state_dict(sd, strict=True)


def train_grpo(cfg, asr=None):
    """cfg: the CLI namespace (cli.py). asr: optional (wav, sr, target_text, lang) -> error rate in [0, 1];
    default Whisper WER against the CSV's `asr_text_col` (one Whisper per scoring worker)."""
    from training.rl.rewards import DNSMOS, SpeakerSim, WhisperASR

    dev = torch.device(cfg.device)
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    full = json.load(open(cfg.config))
    ttl = full["ttl"]
    sr, ccf, ld, ns = full["ae"].get("sample_rate", 44100), ttl["chunk_compress_factor"], ttl["latent_dim"], ttl["normalizer"]["scale"]

    # ---- models: everything frozen but the VF
    te, ref_enc, vf, _, dp, ae_encoder, ae_decoder, mel_spec, hop = build_models(ttl, full["ae"], sr, dev, dp_ckpt_path=cfg.dp_ckpt)
    if dp is None:
        raise FileNotFoundError(f"duration predictor not loaded from {cfg.dp_ckpt}")
    _load_ae(ae_encoder, ae_decoder, cfg.ae_checkpoint)
    base = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    for mod, key in ((te, "text_encoder"), (ref_enc, "reference_encoder"), (vf, "vf_estimator")):
        mod.load_state_dict(base[key], strict=True)
    uncond = None
    if cfg.rollout_cfg != 1.0:                                 # the unconditional tokens the T2L sampler uses
        if base.get("u_text") is None or base.get("u_ref") is None:
            raise ValueError(f"{cfg.ckpt} has no u_text / u_ref: --rollout_cfg {cfg.rollout_cfg} needs them")
        uncond = tuple(base[k].to(dev) if base.get(k) is not None else None for k in ("u_text", "u_ref", "u_key"))
    if cfg.score_workers < 1:
        raise ValueError("--score_workers must be >= 1")
    stats = torch.load(cfg.stats_path, map_location=dev)
    mean, std = stats["mean"].to(dev).view(1, -1, 1), stats["std"].to(dev).view(1, -1, 1)
    for m in (te, ref_enc, vf, dp, ae_encoder, ae_decoder, mel_spec):
        m.eval().requires_grad_(False)
    # KL reference: frozen copies of the starting TE and VF, copied together so a projected style key stays tied.
    te_ref, vf_ref = copy.deepcopy((te, vf))
    vf.requires_grad_(True)
    tied = te.speech_prompted_text_encoder.style_key is not None
    lang_tags = bool(full.get("lang_tags", False) or base.get("lang_tags", 0))
    params = list(vf.parameters())
    print(f"[grpo] VF full fine-tune: {sum(p.numel() for p in params) / 1e6:.2f} M params, tied style key {tied}")
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    n_groups = cfg.prompts_per_iter * cfg.prompt_repeat
    ups = max(1, min(cfg.updates_per_iter, n_groups))
    total = cfg.iters * ups
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / total))   # linear decay to 0
    ema = {k: v.detach().clone() for k, v in vf.state_dict().items()} if cfg.ema > 0 else None
    start = 1
    if cfg.resume:
        rs = torch.load(cfg.resume, map_location=dev, weights_only=False)
        vf.load_state_dict(rs["vf"])
        opt.load_state_dict(rs["opt"])
        sched.load_state_dict(rs["sched"])
        if ema is not None and rs.get("ema"):
            ema = {k: v.to(dev) for k, v in rs["ema"].items()}
        start = rs["iter"] + 1
        print(f"[grpo] resumed {cfg.resume} at iteration {rs['iter']}")

    # ---- prompts and references from the dataset CSV, through the T2L dataset's text pipeline
    ds = Text2LatentDataset(cfg.data, sample_rate=sr, hop_length=hop,
                            affricate_ligatures=bool(full.get("affricate_ligatures", False)))
    df = ds.df
    asrs = queue.Queue()                                       # one ASR model per scoring worker
    if asr is None and cfg.w_asr > 0:
        if cfg.asr_text_col not in df.columns:
            raise ValueError(f"the ASR reward needs the words of each row: no '{cfg.asr_text_col}' column in {cfg.data}")
        for _ in range(cfg.score_workers):
            asrs.put(WhisperASR(cfg.asr_model, cfg.score_device or cfg.device))
    else:
        for _ in range(cfg.score_workers):
            asrs.put(asr)
    targets = df[cfg.asr_text_col].fillna("").astype(str).str.strip().tolist() if cfg.asr_text_col in df.columns \
        else df["text"].tolist()
    sim_fn = SpeakerSim(cfg.score_device or cfg.device)
    mos_fn = DNSMOS(cfg.score_device or cfg.device) if cfg.w_mos > 0 else None
    rng = random.Random(cfg.seed)                             # the reference pool and hard cases: fixed by the seed
    by_spk = {}
    for i, s in enumerate(df["speaker_id"].tolist()):
        by_spk.setdefault(s, []).append(i)
    spks = sorted(by_spk)
    rng.shuffle(spks)
    refs = []
    with torch.no_grad():
        for s in spks:
            n = 0
            for i in rng.sample(by_spk[s], len(by_spk[s])):
                if n >= cfg.ref_pool_per_spk or len(refs) >= cfg.ref_pool:
                    break
                wav = ds._load_wav_by_index(i)
                if not 3.0 <= wav.numel() / sr <= 10.0:
                    continue
                z = compress_latents(ae_encoder(mel_spec(wav[None].to(dev))), factor=ccf)
                z = ((z - mean) / std) * ns
                z_len = torch.tensor([z.shape[2]], device=dev)
                z_ref, ref_mask = build_reference_only(z, z_len, dev, cfg.ref_max_frames)
                z_dp, dp_mask = build_reference_only(z, z_len, dev, cfg.dp_ref_max_frames or cfg.ref_max_frames)
                s_dp = dp.ref_encoder(z_dp, mask=dp_mask).reshape(1, dp.ref_encoder.num_queries, dp.ref_encoder.query_dim)
                refs.append(dict(row=i, style_ttl=ref_enc(z_ref, mask=ref_mask), style_dp=s_dp,
                                 ecapa=sim_fn.embed(wav.numpy(), sr)[0]))
                n += 1
    ref_rows = {r["row"] for r in refs}
    rows = [i for i in range(len(df)) if i not in ref_rows]   # a reference clip is never also a prompt
    if cfg.w_asr > 0:
        rows = [i for i in rows if targets[i]]                 # the WER reward needs each prompt's words
    has_text = cfg.asr_text_col in df.columns                 # the hard cases repeat the same words in the ASR text

    def to_ids(ipa, lang):
        ids = text_to_indices(ipa, lang=lang)
        return wrap_lang_tags(ids, lang) if lang_tags else ids

    prompts_pool = [(df["text"].iloc[i], targets[i], df["lang"].iloc[i]) for i in rows]
    prompts_pool = [p for p in prompts_pool if len(to_ids(p[0], p[2])) <= cfg.max_tokens]
    if not prompts_pool or not refs:
        raise ValueError(f"{len(prompts_pool)} prompts within --max_tokens {cfg.max_tokens} and {len(refs)} 3-10 s "
                         "references")
    print(f"[grpo] {len(prompts_pool)} prompts, {len(refs)} references ({len(spks)} speakers), lang tags {lang_tags}")
    rng = random.Random(cfg.seed + start)

    def pick_prompt():
        for _ in range(100):                                   # a hard case can exceed --max_tokens: draw again
            ipa, target, lang = rng.choice(prompts_pool)
            kind = "row"
            if rng.random() < cfg.hard_frac and len(ipa.split()) >= 3:
                kind, ipa, text = hard_variant(ipa, target if has_text else None, rng.choice(("lwr", "smr", "gsr")), rng)
                target = text if has_text else ipa
            ids = to_ids(ipa, lang)
            if len(ids) <= cfg.max_tokens:
                return dict(ids=ids, target=target, lang=lang, kind=kind, ref=rng.choice(refs))
        ipa, target, lang = rng.choice(prompts_pool)          # every draw too long: a plain prompt (always fits)
        return dict(ids=to_ids(ipa, lang), target=target, lang=lang, kind="row", ref=rng.choice(refs))

    @torch.no_grad()
    def rollout(pr):
        ref = pr["ref"]
        ids = torch.tensor([pr["ids"]], dtype=torch.long, device=dev)
        tmask = torch.ones(1, 1, ids.shape[1], device=dev)
        dur = torch.exp(dp(ids, text_mask=tmask, style_dp=ref["style_dp"], return_log=True)).float().view(1)
        lmask = get_latent_mask((dur * sr).long(), hop, ccf)
        h = te(ids, ref["style_ttl"], text_mask=tmask)
        h_ref = te_ref(ids, ref["style_ttl"], text_mask=tmask)
        n_steps, window = fg.sample_window(rng, cfg.steps_min, cfg.steps_max, cfg.smin_lo, cfg.smin_hi, cfg.sde_steps)
        x, traj = fg.rollout(vf, h, ref["style_ttl"], lmask, tmask, cfg.G, ld * ccf, n_steps, window, cfg.noise_level, gen,
                             cfg.rollout_cfg, uncond)
        z = decompress_latents((x / ns) * std + mean, factor=ccf, target_channels=ld)
        wav = torch.cat([ae_decoder(z[i:i + cfg.decode_bs]).reshape(min(cfg.decode_bs, cfg.G - i), -1)
                         for i in range(0, cfg.G, cfg.decode_bs)]).float().cpu().numpy()   # the whole decoded chunk
        return dict(h=h, h_ref=h_ref, tmask=tmask, lmask=lmask, traj=traj, wav=wav, n_steps=n_steps)

    def score(pr, wav):
        a = asrs.get()                                         # this worker's own ASR model
        try:
            err = np.array([a(w, sr, pr["target"], pr["lang"]) for w in wav]) if a is not None and cfg.w_asr > 0 \
                else np.zeros(len(wav))
        finally:
            asrs.put(a)
        sim = sim_fn(wav, sr, pr["ref"]["ecapa"])
        mos = np.array([mos_fn(w, sr) for w in wav]) if mos_fn else np.zeros(len(wav))
        return err, sim, mos

    def save(it):
        sd = {k: v.detach().cpu().clone() for k, v in (ema or vf.state_dict()).items()}
        ck = {k: v for k, v in base.items() if k not in ("optimizer", "scheduler")}
        ck["vf_estimator"] = sd
        if tied:                                               # the text encoder's key is the trained VF tile
            ck["text_encoder"] = dict(base["text_encoder"], **{"speech_prompted_text_encoder.style_key": sd["tile"]})
        ck["global_step"] = int(base.get("global_step", 0)) + it
        ck["grpo"] = dict(iter=it, base_step=int(base.get("global_step", 0)), ema=cfg.ema, config=vars(cfg))
        path = os.path.join(cfg.out, f"ckpt_step_{ck['global_step']}.pt")
        torch.save(ck, path)
        torch.save(dict(vf=vf.state_dict(), ema=ema, opt=opt.state_dict(), sched=sched.state_dict(), iter=it),
                   os.path.join(cfg.out, "train_state.pt"))
        print(f"[grpo] saved {path}")

    @torch.no_grad()
    def ema_update():
        for k, v in vf.state_dict().items():
            if v.is_floating_point():
                ema[k].mul_(cfg.ema).add_(v, alpha=1.0 - cfg.ema)
            else:
                ema[k].copy_(v)

    gen = torch.Generator(device=dev).manual_seed(cfg.seed + start)
    log = open(os.path.join(cfg.out, "grpo_log.jsonl"), "a")
    pool = ThreadPoolExecutor(max_workers=cfg.score_workers)  # groups are scored while the next ones roll out
    for it in range(start, cfg.iters + 1):
        t0 = time.time()
        prompts = [pr for pr in (pick_prompt() for _ in range(cfg.prompts_per_iter)) for _ in range(cfg.prompt_repeat)]
        rolls, futs = [], []
        for pr in prompts:                                     # Eq. 12 normalises over the whole iteration
            ro = rollout(pr)
            futs.append(pool.submit(score, pr, ro.pop("wav")))
            rolls.append(ro)
        scores = [f.result() for f in futs]
        err, sim, mos = (np.stack([s[j] for s in scores]) for j in range(3))
        adv, _ = fg.advantages(sim, err, mos, cfg.w_sim, cfg.w_asr, cfg.w_mos)
        batch = list(zip(prompts, rolls, adv))
        st = dict(loss=0.0, kl=0.0, clipfrac=0.0, ratio=0.0, n=0, skipped=0)
        gn = torch.zeros(())
        for chunk in (batch[i::ups] for i in range(ups)):     # later chunks are slightly off-policy
            opt.zero_grad(set_to_none=True)
            used = 0
            for pr, ro, a in chunk:
                if np.allclose(a, 0):                          # zero-std group: discarded
                    st["skipped"] += 1
                    continue
                s = fg.ppo_backward(vf, vf_ref, ro["h"], pr["ref"]["style_ttl"], ro["lmask"], ro["tmask"], ro["traj"],
                                    a, cfg.clip, cfg.beta, cfg.noise_level, cfg.micro_batch, 1.0 / len(chunk),
                                    h_ref=ro["h_ref"], adv_clip=cfg.adv_clip, cfg=cfg.rollout_cfg, uncond=uncond)
                used += 1
                for k in ("loss", "kl", "clipfrac", "ratio"):
                    st[k] += s[k]
            st["n"] += used
            if used:                                           # no step when every group of the chunk was discarded
                gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step()
                if ema is not None:
                    ema_update()
            sched.step()
        n = max(1, st["n"])
        rec = dict(it=it, sec=round(time.time() - t0, 1), err=float(err.mean()), sim=float(sim.mean()), mos=float(mos.mean()),
                   loss=st["loss"] / n, kl=st["kl"] / n, clipfrac=st["clipfrac"] / n, ratio=st["ratio"] / n,
                   grad_norm=float(gn), lr=opt.param_groups[0]["lr"], skipped=st["skipped"],
                   kinds=[p["kind"] for p in prompts], n_steps=[r["n_steps"] for r in rolls])
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[grpo] it {it}: err {rec['err']:.4f} sim {rec['sim']:.4f} mos {rec['mos']:.2f} loss {rec['loss']:+.2e} "
              f"kl {rec['kl']:.2e} ratio {rec['ratio']:.5f} clip {rec['clipfrac']:.2f} gn {rec['grad_norm']:.2e} "
              f"lr {rec['lr']:.2e} skip {rec['skipped']} {rec['sec']} s", flush=True)
        if it % cfg.save_every == 0 or it == cfg.iters:
            save(it)
    pool.shutdown()
    log.close()
