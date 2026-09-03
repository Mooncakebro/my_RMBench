# CompactMoDE

COMPACT-2B memory-augmented VLM + MoDE DiT diffusion action head for RMBench
dual-arm manipulation. See `IDEA.md` for the locked design decisions — read it
before changing anything.

## Architecture

```
per frame t:  head-cam RGB (240x320) + instruction + state(14) + prev_action(14)
│
├─ Qwen3-VL-2B (vision encoder + 28-layer decoder)   [trainable; FREEZE_BASE=1 freezes]
│    └─ SideMemoryModule ×28   [trainable, COMPACT variant only]
│         predict(FiLM-GRU, ctrl=prev-action MLP) → layer → observe → Kalman correct → inject
│         state (M: 16×512, P) carried across frames; TBPTT chunk_size=8, detach at boundaries
│
├─ ConditioningBridge  [trainable]
│    image-token hidden states → avg-pool 32 → Linear → state_images (B,32,2048)
│    instruction-token hidden states → masked mean → Linear → goal (B,1,512)
│
└─ MoDeDiT (StateTokenMoDeDiT)  [trainable ~100M @ 1024d/6L]
     EDM diffusion, tokens [σ | goal | proprio(14) | 32 image tokens | noisy actions]
     → action chunk (B,10,14), normalized; decode: z-score⁻¹ joints, minmax⁻¹ grippers
```

| Module | Params | Frozen? |
|---|---|---|
| Qwen3-VL-2B base | ~2.1B | trainable by default; `FREEZE_BASE=1` freezes |
| Side memory ×28 (compact only) | ~300M | trainable (own LR 5e-6) |
| prev-action MLP (compact only) | 0.5M | trainable |
| ConditioningBridge | 5.3M | trainable |
| MoDeDiT head | ~100M (8.4M in debug config) | trainable |

Losses: EDM diffusion MSE (both variants) + COMPACT aux L_obs/L_nll/mem-L2
(compact variant only; no text CE — the policy never generates text).

## Layout

```
policy/CompactMoDE/
├── IDEA.md                      # design decisions (read first)
├── compact_mode/                # package: config, models, bridge, dataset, losses,
│                                #   normalizer, compact_wrapper (copied COMPACT forward),
│                                #   modedit_ext (multi-token DiT subclass)
├── train_baseline.py            # Qwen3-VL-2B + DiT, random-frame batches
├── train_compact.py             # COMPACT+MoDE, episode-sequential TBPTT
├── shell/                       # env-var-driven launchers (JAMEL-COMPACT style)
├── debug/                       # smoke + overfit wiring tests
├── scripts/hdf5_to_lerobot_v3.py# RMBench HDF5 -> LeRobot v3 converter (already run)
├── deploy_policy.py/.yml        # RMBench eval contract
└── runs/                        # checkpoints (created by training)
```

## Environment

```bash
source /home/spc/anaconda3/etc/profile.d/conda.sh && conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # libstdc++ workaround
export HF_HUB_OFFLINE=1                                     # Qwen3-VL-2B is cached
unset ALL_PROXY all_proxy                                   # broken socks proxy breaks HF
```

(The `shell/*.sh` and `debug/run_smoke.sh` scripts do all of this for you.)
If a download is ever needed: `HF_ENDPOINT=https://hf-mirror.com` (huggingface.co
is unreachable through the local proxy).

## Data prep (already done — 12 tasks in `../../data_lerobot/`)

```bash
python scripts/hdf5_to_lerobot_v3.py --tasks swap_blocks            # one task
python scripts/hdf5_to_lerobot_v3.py                                # all 12 tasks
# source: /media/spc/新加卷/RMBench/data/<task>/demo_clean (symlinked at ../../data/)
```

## Debug / wiring tests (run on this dev PC, 8GB GPU)

```bash
bash debug/run_smoke.sh                       # both variants: shapes, backward, memory carry (~2 min)
python debug/overfit_single_episode.py --variant baseline --steps 200
python debug/overfit_single_episode.py --variant compact  --steps 100
python debug/test_deploy.py                   # save_pretrained -> CompactMoDEDeployer -> fake obs
```

Overfit = train on 100 frames of `swap_blocks` episode 0; the diffusion loss
should drop well below 50% of its initial value (measured: baseline 1.91→0.11,
compact 2.04→0.43). The compact debug runs use SGD for the 300M side-memory
params (`--mem-opt sgd`; AdamW's m/v states OOM this GPU — server training uses
AdamW everywhere, which is the default in `train_compact.py`).

Mini end-to-end training-loop checks (checkpoints + save_pretrained):

```bash
TASK=swap_blocks FREEZE_BASE=1 GRAD_CKPT=1 MAX_STEPS=3 BATCH_SIZE=4 \
  NUM_WORKERS=0 EMBED_DIM=256 N_LAYERS=2 N_HEADS=4 SAVE_STEPS=2 \
  OUTPUT_DIR=$(pwd)/runs/debug_baseline bash shell/run_train_baseline.sh
python train_compact.py --task swap_blocks --freeze-base 1 --grad-ckpt 1 \
  --max-steps 2 --chunk-size 4 --num-streams 1 --embed-dim 256 --n-layers 2 \
  --n-heads 4 --mem-opt sgd --max-frames-per-episode 32 --save-steps 1 \
  --output-dir $(pwd)/runs/debug_compact
```

## Training (8×A800 server)

Baseline (stateless Qwen3-VL-2B + DiT):

```bash
TASK=swap_blocks BATCH_SIZE=16 MAX_STEPS=30000 \
  bash shell/run_train_baseline.sh
```

COMPACT+MoDE (memory, TBPTT):

```bash
TASK=swap_blocks NUM_STREAMS=8 CHUNK_SIZE=8 MAX_STEPS=10000 \
  bash shell/run_train_compact.sh
```

Useful env vars (both scripts): `FREEZE_BASE=1` (freeze Qwen), `GRAD_CKPT=1`,
`LR` (bridge+DiT, default 1e-4), `BASE_LR` (Qwen, 1e-5), `MEMORY_LR` (compact
only, 5e-6), `EMBED_DIM/N_LAYERS/N_HEADS/NUM_EXPERTS/TOP_K` (DiT size),
`MEM_DIM` (COMPACT memory feature dimension, default 512), `DATA_ROOT`,
`OUTPUT_DIR`, `SAVE_STEPS`, `TASK`.

Notes:
- One optimizer step of `train_compact.py` = one chunk = NUM_STREAMS × CHUNK_SIZE
  VLM forwards (every frame, stride 1). 10k steps × 8 streams × 8 frames covers
  ~12 epochs of swap_blocks.
- Scripts are single-process. For multi-GPU, launch one process per GPU with
  different `CUDA_VISIBLE_DEVICES` + `TASK`, or wrap in torchrun (DDP wiring is
  not implemented yet — the training loops are deliberately simple).
- Checkpoints: only `best/` and `final/` deployment-format directories are saved;
  `best/` is selected from periodic training-loss checks (`BEST_SAVE_STEPS`,
  default 500) and `final/` is the last step. `training_summary.json` records
  the best step/loss. Periodic optimizer
  checkpoints are disabled to limit disk usage; new runs cannot be resumed
  unless an older `.pt` checkpoint is supplied explicitly with `--resume`.

## Eval (RMBench sim, run on the eval machine)

```bash
cd /home/spc/memory_arena/RMBench
python script/eval_policy.py --config policy/CompactMoDE/deploy_policy.yml \
  --overrides --task_name swap_blocks --task_config demo_clean \
  --checkpoint_path policy/CompactMoDE/runs/compact_swap_blocks/final \
  --variant compact
```

(`variant` ∈ {compact, baseline}; `checkpoint_path` = a `save_pretrained` dir.
The policy predicts a 10-step chunk but executes only its first action, then
re-observes and replans. Compact memory therefore updates once per executed
environment step.)

## Known issues / gotchas

- **Baseline weight loading**: do not load Qwen3-VL via `AutoModel` — checkpoint
  keys carry the `model.` prefix and the bare `Qwen3VLModel` silently stays
  random. We load `AutoModelForImageTextToText` and call `.model` (skips lm_head).
- `MoDeDiT.forward` assumes a single state token; multi-token state positions
  come from `compact_mode/modedit_ext.StateTokenMoDeDiT` (MoDE repo is not modified).
- JAMEL-COMPACT's forward doesn't expose hidden states/aux losses without text
  labels; `compact_mode/compact_wrapper.CompactModeWrapper` is a subclass with
  a copied, adapted forward (also skips the LM head entirely). Neither the
  MoDE repo nor the JAMEL-COMPACT repo is modified in any way.
- The hydra DiT config must keep the typo'd key `embed_pdrob` verbatim.
- transformers 4.57.6 quirks handled in code: `get_image_features` returns a
  plain tuple; the processor never emits `mm_token_type_ids` (we build it).
- `torch_dtype` deprecation warning from transformers is harmless.

## Trouble shooting
### Insatll Env for COMPACT-MoDE Trainig:
```bash
conda create -y -n lerobot python=3.10                    # 创建一个名为lerobot的python3.10环境
conda activate lerobot                                    # 激活名为lerobot的python3环境
conda install ffmpeg -c conda-forge                       # conda安装ffmpeg库，不装很多示例会直接报错。
pip install lerobot                                       # 安装lerobot，从库中安装，不从源码安装
pip install "lerobot[aloha]"                              # 安装lerobot额外的依赖 alpha 模拟环境
pip install "lerobot[metaworld]"                          # 安装lerobot额外的依赖 metaworld 基准测试
pip install "lerobot[feetech]"                            # 安装lerobot额外的依赖 feetech
pip install "lerobot[smolvla]"                            # 安装lerobot额外的依赖 smolvla

pip install omegaconf hydra-core
```

### Insatll Env for Eval:
```bash
conda create -n RMBench python=3.10 -y

conda activate RMBench

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

bash script/_install_tsinghuayuan.sh

'''
if fail to install pytorch3d and curobo, do:

git clone --branch stable https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d
cd /tmp/pytorch3d
pip install -e . --no-build-isolation

# 手动 clone（指定 tag 和浅克隆）
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git

# 确认目录存在后再进入
cd curobo
pip install -e . --no-build-isolation
cd ../..
'''

python -m pip install \
    "transformers==4.57.6" \
    accelerate \
    qwen-vl-utils \
    "hydra-core==1.1.1" \
    "omegaconf==2.1.2" \
    pandas \
    pyarrow \
    av

pip install setuptools==69.5.1

python -m pip install ninja einops "h5py==3.16.0"

python -m pip install --force-reinstall "setuptools<81"



```