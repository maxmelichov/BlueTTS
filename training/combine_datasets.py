import argparse
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.text_vocab import CHAR_TO_ID

COLUMNS = ["filename", "whisper_phonemes", "speaker_id", "wer_score", "lang", "phonemized"]
VALID_CHARS = set(CHAR_TO_ID.keys())
DEFAULT_ESPEAK_LANG_MAP = {"en": "en-us", "es": "es", "de": "de", "fr": "fr"}


def load_csv(spec):
    path = spec["csv"]
    if not os.path.exists(path):
        print(f"  skip: {path} not found")
        return []

    df = pd.read_csv(path, **spec.get("csv_kwargs", {}))
    print(f"  {os.path.basename(path)}: {len(df)} rows")

    if (fc := spec.get("filter_col")) and fc in df.columns:
        df = df[df[fc].astype(bool)]

    fn = spec.get("filename_col", "filename")
    if (strip := spec.get("strip_prefix")) and fn in df.columns:
        df[fn] = df[fn].str.replace(strip, "", regex=False)
    if tpl := spec.get("filename_template"):
        df[fn] = df.apply(lambda r: tpl.format(**r.to_dict()), axis=1)
    if (ad := spec.get("audio_dir")) and fn in df.columns:
        def _prefix(x, ad=ad):
            x = str(x).strip()
            return x if os.path.isabs(x) or x.startswith(ad) else os.path.join(ad, x)
        df[fn] = df[fn].apply(_prefix)
    if fn != "filename" and fn in df.columns:
        df = df.rename(columns={fn: "filename"})

    tc = spec.get("text_col") or next((c for c in ("whisper_phonemes", "text", "phonemes") if c in df.columns), None)
    if tc and tc != "whisper_phonemes":
        df = df.rename(columns={tc: "whisper_phonemes"})

    df["lang"] = spec.get("lang", "he")
    for k, v in spec.get("extra_cols", {}).items():
        df[k] = v

    if splits := spec.get("splits"):
        out = []
        for s in splits:
            sub = df[df["filename"].str.contains(s["pattern"], regex=False)].copy()
            if sub.empty:
                continue
            sub["speaker_id"] = s["speaker_id"]
            out.append(sub)
        return out

    df["speaker_id"] = spec["speaker_id"]
    return [df]


def load_libritts(spec):
    base = spec["base_dir"]
    if not os.path.exists(base):
        return []
    offset, lang = spec.get("speaker_offset", 0), spec.get("lang", "en")
    rows = []
    for split in spec.get("splits", []):
        for root, _, files in os.walk(os.path.join(base, split)):
            for fn in files:
                if not fn.endswith(".normalized.txt"):
                    continue
                wav = os.path.join(root, fn[:-15] + ".wav")
                if not os.path.exists(wav):
                    continue
                try:
                    spk = offset + int(os.path.basename(os.path.dirname(root)))
                except ValueError:
                    continue
                text = open(os.path.join(root, fn)).read().strip()
                if text:
                    rows.append({"filename": wav, "whisper_phonemes": text,
                                 "speaker_id": spk, "wer_score": 0.0, "lang": lang})
    if not rows:
        return []
    df = pd.DataFrame(rows)
    print(f"  LibriTTS: {len(df)} rows, {df['speaker_id'].nunique()} speakers")
    return [df]


LOADERS = {"csv": load_csv, "libritts": load_libritts}


def _split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _direct_config(args):
    datasets = []
    if args.libritts:
        datasets.append(
            {
                "type": "libritts",
                "base_dir": args.libritts,
                "splits": _split_csv(args.splits),
                "speaker_offset": args.speaker_offset,
                "lang": args.lang,
            }
        )
    if args.csv:
        if args.speaker_id is None:
            raise SystemExit("--speaker-id is required when using --csv")
        spec = {
            "csv": args.csv,
            "speaker_id": args.speaker_id,
            "lang": args.lang,
        }
        if args.audio_dir:
            spec["audio_dir"] = args.audio_dir
        if args.filename_col:
            spec["filename_col"] = args.filename_col
        if args.text_col:
            spec["text_col"] = args.text_col
        datasets.append(spec)
    if not datasets:
        raise SystemExit("Pass --config, or use --libritts/--csv to build a dataset directly.")
    return {
        "output": args.output or "combined_dataset.csv",
        "clean_output": args.clean_output or "combined_dataset_cleaned.csv",
        "espeak_lang_map": DEFAULT_ESPEAK_LANG_MAP,
        "datasets": datasets,
    }


def combine(config, output, limit=None):
    dfs = []
    for spec in config.get("datasets", []):
        loader_name = spec.get("type", "csv")
        if loader_name not in LOADERS:
            raise ValueError(f"Unknown dataset type: {loader_name}")
        dfs.extend(LOADERS[loader_name](spec))
    if not dfs:
        return None
    for df in dfs:
        if "wer_score" not in df.columns:
            df["wer_score"] = 0.0
        if "lang" not in df.columns:
            df["lang"] = "he"
    combined = pd.concat([df[[c for c in COLUMNS if c in df.columns]] for df in dfs], ignore_index=True)
    if limit is not None:
        combined = combined.head(limit)
    combined.to_csv(output, index=False)
    print(f"Combined → {output} ({len(combined):,} rows)")
    return output


def validate(text, lang):
    if not isinstance(text, str) or not text.strip():
        return "", False
    text = text.strip().strip('"')
    if lang == "he":
        text = text.replace("g", "ɡ").replace("r", "ʁ")
    if set(text) - VALID_CHARS:
        return text, False

    words = text.split()
    if len(words) >= 3 and any(words[i].lower() == words[i+1].lower() == words[i+2].lower()
                                for i in range(len(words) - 2)):
        return text, False
    return text, True


def clean(input_file, output_file):
    df = pd.read_csv(input_file)
    results = [validate(t, l) for t, l in zip(df["whisper_phonemes"], df.get("lang", ["he"] * len(df)))]
    df["whisper_phonemes"] = [r[0] for r in results]
    df = df[[r[1] for r in results]].copy()
    df.to_csv(output_file, index=False)
    print(f"Cleaned → {output_file} ({len(df):,} rows)")


def phonemize(input_file, espeak_map=None):
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    from data.text_vocab import normalize_text
    from tqdm import tqdm

    espeak_map = espeak_map or DEFAULT_ESPEAK_LANG_MAP
    df = pd.read_csv(input_file)

    todo = df["lang"] != "he"
    if "phonemized" in df.columns:
        todo &= ~df["phonemized"].fillna(False)
    if not todo.any():
        print("[Phonemize] nothing to do")
        return

    sep = Separator(phone="", word=" ", syllable="")
    for lang in df.loc[todo, "lang"].unique():
        mask = todo & (df["lang"] == lang)
        texts = df.loc[mask, "whisper_phonemes"].str.replace(r'["\u201c\u201d]', "", regex=True).tolist()
        espeak_lang = str(espeak_map.get(lang, lang))
        backend = EspeakBackend(espeak_lang, preserve_punctuation=True,
                                with_stress=True, language_switch="remove-flags")
        ipa = []
        for i in tqdm(range(0, len(texts), 1000), desc=f"espeak ({lang})"):
            ipa.extend(backend.phonemize(texts[i:i+1000], separator=sep, njobs=os.cpu_count() or 1))
        df.loc[mask, "whisper_phonemes"] = [normalize_text(t) for t in ipa]
        df.loc[mask, "phonemized"] = True

    df.loc[df["lang"] == "he", "phonemized"] = True
    df.to_csv(input_file, index=False)
    print(f"[Phonemize] Saved → {input_file}")


def _print_next_steps(clean_out):
    print("\nNext steps:")
    print(f"  1. Compute stats for this exact CSV:")
    print(f"     uv run python compute_latent_stats.py --metadata {clean_out} --output runs/my_dataset/stats_multilingual.pt")
    print("  2. Train DP with that stats file:")
    print(f"     uv run python -m training.dp.cli --data {clean_out} --stats_path runs/my_dataset/stats_multilingual.pt")
    print("  3. Train T2L with the same stats file:")
    print(f"     uv run python -m training.t2l.cli --data {clean_out} --stats_path runs/my_dataset/stats_multilingual.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Combine training metadata. Either pass --config datasets.json, or use "
            "direct flags like --libritts /data/LibriTTS_R --splits train-clean-360."
        )
    )
    p.add_argument("--config", help="JSON config with a datasets list")
    p.add_argument("--output")
    p.add_argument("--clean-output")
    p.add_argument("--limit", type=int, help="Keep only the first N rows for smoke testing")
    p.add_argument("--skip-combine", action="store_true")
    p.add_argument("--skip-clean", action="store_true")
    p.add_argument("--skip-phonemize", action="store_true")
    p.add_argument("--no-next-steps", action="store_true")

    direct = p.add_argument_group("direct dataset flags")
    direct.add_argument("--libritts", help="LibriTTS/LibriTTS_R root containing split folders")
    direct.add_argument("--splits", default="train-clean-100,train-clean-360")
    direct.add_argument("--speaker-offset", type=int, default=1000)
    direct.add_argument("--csv", help="Generic metadata CSV")
    direct.add_argument("--audio-dir", help="Prefix relative audio paths for --csv")
    direct.add_argument("--speaker-id", type=int, help="Speaker id for a single-speaker --csv")
    direct.add_argument("--filename-col", help="CSV column containing audio path/name")
    direct.add_argument("--text-col", help="CSV column containing text/phonemes")
    direct.add_argument("--lang", default="he")
    args = p.parse_args()

    config = json.load(open(args.config)) if args.config else _direct_config(args)
    out = args.output or config.get("output", "combined_dataset.csv")
    clean_out = args.clean_output or config.get("clean_output", "combined_dataset_cleaned.csv")

    if not args.skip_combine:
        combine(config, out, limit=args.limit)
    if not args.skip_clean:
        clean(out, clean_out)
    if not args.skip_phonemize:
        phonemize(clean_out, config.get("espeak_lang_map"))
    if not args.no_next_steps:
        _print_next_steps(clean_out)
