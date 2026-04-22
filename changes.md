● Senior-engineer review — Light-BlueTTS                                                    
                                                                                            
  The code works and the README is friendly, but the internals carry too much duplication   
  and a couple of god-files. Here's what I'd actually do, ordered by ROI.                   
                                                                                            
  ---                                                                                       
  1. Inference — kill the three-way duplication                                             
                                                                                            
  src/blue_onnx/__init__.py, src/blue_pt/__init__.py, src/blue_trt/__init__.py reimplement
  the same pipeline (_load_style_json, _predict_duration, _flow_matching, _apply_fade,      
  synthesize, chunking) with 85–95% overlap. Every bug fix has to be made three times.      
                                                                                            
  Refactor:                                                                                 
  src/blue_tts/                                                                             
    core/                                                                                   
      pipeline.py        # synthesize(), chunking, fade, silence stitching                  
      duration.py        # _predict_duration with the heuristic (lines 205-209)
      flow_matching.py   # _flow_matching loop (sampler-agnostic)                           
      style.py           # _load_style_json + Style dataclass                               
      config.py          # dataclass schema for tts.json (replace .get().get().get())       
    backends/                                                                               
      base.py            # Backend Protocol: text_encoder, dp, vf, vocoder, ref_encoder     
      onnx_backend.py    # 80 lines: only session loading + forward()                  
      pt_backend.py      # 80 lines                                                         
      trt_backend.py     # 120 lines (TRTEngine wrapper stays here)                         
    text/                                                                                   
      processor.py       # was _common.py                                                   
      vocab.py           # was _blue_vocab.py
                                                                                            
  Public API stays BlueTTS(backend="onnx"|"pt"|"trt", ...). Each backend is a thin adapter  
  that exposes 4 callables. Pipeline code lives once.                                       
                                                                                            
  2. Config — stop scattering magic numbers                                                 
                  
  - src/blue_onnx/__init__.py:56-67 re-defines defaults per-backend with cfg.get("ttl",     
  {}).get(...). Replace with a pydantic/dataclass schema loaded once and passed to backends.
  - Hardcoded heuristics: T_lat = int(text_ids.shape[1] * 1.3) and the (20, 3, 600, 800) cap
   at blue_onnx/__init__.py:205-209 — promote to DurationConfig(text_to_lat_ratio=1.3,      
  min_lat=20, char_factor=3, abs_cap=800).
  - TARGET = 256 at blue_trt/__init__.py:275 should come from config.                       
  - voices/*.json has no schema. Add VoiceStyle dataclass with from_json validation.        
                                                                                            
  3. Training — currently the hardest thing in the repo                                     
                                                                                            
  training/src/train_text_to_latent.py is 1298 lines doing data loading, model init,        
  CFG-param extraction, training loop, validation, checkpointing. To make this easier for 
  users:                                                                                    
                  
  - Split into modules:                                                                     
  training/
    data_module.py     # datasets, collate, samplers                                        
    builders.py        # build_models(cfg) → (text_enc, vf, dp, ref_enc, vocoder)           
    trainer.py         # the loop only — step(), eval_step(), save()                        
    cfg_utils.py       # u_text/u_ref CFG param logic                                       
    cli.py             # argparse + entry point                                             
  - Replace sys.path.append at training/src/train_text_to_latent.py:9 and                   
  src/blue_pt/__init__.py:14-17 — these are the most fragile thing in the repo. Make        
  training a real package (training/__init__.py), declare it in pyproject.toml, and import
  normally.                                                                                 
  - Single command to start training: uv run blue-train --config config/tts.json --data 
  data/train.csv --out runs/exp1. Right now the entry point is unclear and there's no       
  example CSV / data layout doc.
  - Add a one-page training/README.md: dataset format (CSV columns, expected sample rate), a
   tiny data/sample/ with 3 rows, what each config knob does, what a healthy loss curve     
  looks like.
  - Consider lifting the loop onto 🤗 Accelerate or PyTorch Lightning. You'd delete ~400    
  lines of DDP/checkpoint/AMP boilerplate.                                                  
  
  4. Examples — collapse 5 near-identical files                                             
                  
  examples/{hebrew,english,spanish,italian,german}.py are 15 lines each, only text and lang 
  change. Delete them. Keep examples/app.py as the canonical CLI and add a
  examples/voices.md table showing one-liner invocations:                                   
                  
  uv run python -m blue_tts --lang he --voice female1 --text "םולש"
  uv run python -m blue_tts --lang en --voice female1 --text "Hello"                        
  
  5. Scripts — split export_onnx.py (614 lines)                                             
                  
  Currently mixes graph surgery, the OnnxSafeMultiheadAttention class (~70 lines),          
  state-dict loading, and quantization. Split into scripts/export/{builder.py, attention.py,
   quantize.py, cli.py}. Same medicine for create_tensorrt.py.                              
                  
  6. Cross-cutting hygiene

  - Replace print("[INFO] ...") (blue_trt/__init__.py:211, 224, blue_onnx/_common.py:56,    
  convert_stats.py) with logging.getLogger(__name__). Five-minute change, big quality bump.
  - Type hints: training functions like phonemize_text, extract_style, encode_wav_to_latent 
  have no annotations. Add them — IDE navigation will improve immediately.                  
  - Naming: T_lat, T_cap, h_text, v_cond, z_pred are cryptic. Rename to latent_frames,
  latent_frames_cap, text_hidden, velocity_cond, predicted_latent. Cost: 30 min with        
  rename-symbol; payoff: every future reader.
  - Dead code: blue_trt/__init__.py:68 idx = self.engine.get_tensor_name is unused;         
  tensor-conversion at 442-443 is in a hot loop and should be hoisted.                      
  - JSON config has no comments. Either move tts.json to tts.yaml (comments allowed) or
  generate a tts.config.md describing every key.                                            
                  
  7. Suggested final layout                                                                 
                  
  Light-BlueTTS/                                                                            
  ├── src/blue_tts/          # one package, three backends behind it
  │   ├── core/  backends/  text/                                                           
  │   └── __init__.py        # re-exports BlueTTS
  ├── training/              # real package, no sys.path tricks                             
  ├── scripts/export/  scripts/trt/                                                         
  ├── examples/              # one CLI + a markdown cookbook                                
  ├── config/tts.yaml        # commented; validated by core/config.py                       
  └── docs/                                                                                 
      ├── inference.md  training.md  voices.md                                              
                  
  ---                                                                                       
  Where I'd start (1 week of work, in order)
                                                                                            
  1. Day 1 — extract Backend Protocol + move shared synthesize/_flow_matching to
  core/pipeline.py. Pure refactor, no behaviour change. Add a smoke test that runs all three
   backends on the same prompt and diffs output length.
  2. Day 2 — config dataclass + remove magic numbers. Delete the 4 backend _load_config     
  methods.                                                                                  
  3. Day 3 — kill the sys.path hacks; make training an installable package.
  4. Day 4 — split train_text_to_latent.py into 5 modules; write training/README.md + sample
   data.                                                                                    
  5. Day 5 — collapse example scripts, swap print for logging, rename cryptic vars, split   
  export_onnx.py.                                                                           
                  
  Want me to start on step 1 (the backend protocol + shared pipeline)? That's the change    
  that pays back the most for the least risk.