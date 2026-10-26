# Mem0-Compact v3: Long-History Context With COMPACT Fusion

## Goal

v3 tests whether a compact set of visual observations spanning a long interval
helps action prediction, without retaining an autograd graph across time. It
combines Mem-0's useful separation between an episode anchor and a recent
sliding window with COMPACT's per-layer Predict/Correct/Inject modules.

The first implementation is an opt-in training variant. Existing Mem0-Compact
configs have no `history_context.enabled` setting and keep their recurrent
COMPACT/TBPTT behavior unchanged.

## Input and Model Semantics

At an episode position `t`, v3 builds one multimodal prompt containing:

1. The episode's initial image as a persistent anchor, when enabled.
2. `num_history_frames` images selected approximately evenly from the previous
   `lookback_seconds` interval, excluding the current frame.
3. The current image.

The v3 config defaults to a 20-second look-back at 30 fps. That is a span of
about 600 dataset frames. With four sampled history frames and the initial and
current images, a mature context contains up to six images. Early in an episode
there are fewer unique frames. Duplicate indices are removed.

All images are passed to Qwen3-VL in chronological order as visual content in
one prompt. The current instruction is used for the prompt. COMPACT's existing
per-layer observation pooling sees the visual tokens for the whole prompt and
uses its learned observation queries, Kalman-style correction, and gated
injection to fuse that multi-image evidence. The action head receives the
resulting current summary and current proprioceptive state.

This is not a learned dynamics function `f(s_t, a_t) = s_(t+1)`. The policy
still predicts an action chunk conditioned on visual context and state. The
historical images give that prediction access to sparse past evidence.

## COMPACT State and TBPTT

For each training example, COMPACT starts from its learned initial per-layer
memory state and processes the full image context in one forward. That memory
is not carried to the next dataset sample. The action and auxiliary losses
backpropagate through the current multi-image forward, but not through previous
or future samples.

The existing trainer's `window_size` therefore means the number of independent
examples accumulated into an optimizer update in v3; it no longer determines
the length of a recurrent graph. The trainer backpropagates each independent
sample immediately and uses DDP `no_sync()` on intermediate samples. `grad_accum_windows`
continues to multiply the number of examples per optimizer update.

This removes long temporal BPTT while retaining gradient flow through COMPACT
within each current context. It does not remove ordinary activation memory,
and it does not guarantee a faster iteration: the VLM now encodes multiple
images per example.

## History Sampling

`RandomEpisodeIterableDataset` still yields current frames sequentially within
an episode. In history mode, it creates a per-sample image list by looking up
the episode's initial index, the selected past indices, and the current index.
The image cache is bounded to the configured look-back interval plus the
initial anchor, so previously decoded images can be reused without retaining
an entire long episode in worker memory.

History never crosses episode boundaries. The current frame remains the anchor
for the action target: the dataset's action chunk starts at `t`, is capped by
the configured action horizon, and uses the existing LeRobot normalization.
The validation loader uses the same history construction but disables image
augmentation.

## Mem-0 Reference

Official Mem-0 stores the first image feature as a one-item anchor and keeps a
FIFO window of recent image features (`window_size: 30`). Its M1 dataset is
30 fps, so that FIFO covers about one second. Mem-0 performs feature-level
cross-attention between the current visual feature and its anchor/window.

v3 borrows the anchor-plus-recent-history selection idea, but does not copy the
Mem-0 MemoryBank. Selected raw frames are encoded together by Qwen, then fused
through COMPACT's per-layer observation and correction modules. It is a
different compute/memory tradeoff: several visual encodings per action example
instead of one new encoding and a recurrent feature cache.

## Configuration

The starting config is `source/config/mem0_compact_v3_train.yaml`:

```yaml
history_context:
  enabled: true
  lookback_seconds: 20.0
  fps: 30
  num_history_frames: 4
  include_initial_frame: true
```

The frame span is `round(lookback_seconds * fps)`. The default assumes the
dataset's 30 fps metadata. If a dataset has another rate, update `fps` to match
its `meta/info.json`. `num_history_frames` counts only samples from the recent
interval; it excludes the initial anchor and current observation.

For a cold Qwen start, do not pass `--init-from-mem0`; pass the local base model
path and use the v3 config. For a warm start, add the released Mem-0 checkpoint
with `--init-from-mem0`. A Compact checkpoint stores the resolved config,
including v3 history settings, and the deployment agent uses that setting to
build the same context. Existing v2 checkpoints/configs remain on the recurrent
path.

Example launch from the RMBench root (edit server-local model/checkpoint paths):

```bash
CONFIG="$REPO_ROOT/policy/Mem0-Compact/source/config/mem0_compact_v3_train.yaml" \
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=29502 NPROC=4 \
TASK=swap_blocks BATCH_SIZE=1 GRAD_ACCUM_WINDOWS=1 MAX_STEPS=30000 \
FREEZE_BASE=1 \
OUTPUT_DIR="$REPO_ROOT/policy/Mem0-Compact/runs/compact_v3_qwen_only" \
SAVE_BEST=1 BEST_START_STEP=1000 BEST_MIN_DELTA=0.0001 \
SAVE_EVERY_STEPS=2000 SAVE_FINAL=1 \
EXTRA_ARGS="--model-path $BASE_MODEL --window-size 8 --grad-ckpt 1 --num-workers 1" \
bash policy/Mem0-Compact/source/training/train_ddp.sh \
2>&1 | tee "$REPO_ROOT/train_compact_v3_qwen_only.log"
```

## Outputs and Evaluation

Training checkpoints use the existing `ckpt_best.pt`, numbered checkpoint,
and `ckpt_final.pt` formats. For open-loop evaluation, use the existing
`script/open_loop_eval_compact.py --policy mem0` command with the v3 checkpoint.
It reads `history_context` from the checkpoint config, maintains the same
initial-plus-spaced-frame input, and reinitializes COMPACT memory per frame.
Closed-loop evaluation reconstructs the history setting from the checkpoint
config and maintains a bounded per-episode image history in the agent.

Open-loop metrics are teacher-forced: observations always come from the dataset,
not from the policy's predicted actions. They measure action prediction under
ground-truth observation histories, not closed-loop task success.

## Costs and Limitations

- The initial anchor is re-encoded on each sample; the first implementation
  caches decoded dataset images, not Qwen visual features. This preserves a
  straightforward gradient path when Qwen is trainable but adds substantial
  VLM compute even when Qwen is frozen.
- Six images per mature example can increase visual tokens, attention memory,
  data decode work, and wall-clock time. Start with a short smoke run and
  inspect GPU memory and the trainer's data/forward/backward profile.
- The model pools the multi-image visual token set into COMPACT observations;
  explicit per-frame temporal embeddings beyond Qwen's token order are not
  added. The current image appears last, but the observation pooling is learned.
- Training samples at 30 fps, while closed-loop evaluation may call the policy
  at a different control frequency. `fps` controls dataset index spacing; it
  does not resample online observations to an exact wall-clock cadence. Confirm
  the environment's policy-call rate before interpreting 20 seconds literally.
- The historical proprioceptive states are not separately appended. The anchor
  and sampled history are visual; current proprioception remains the action
  head's state input, matching the way Mem-0's anchor is a stored visual
  feature rather than a complete raw observation.
- This v3 variant currently targets the COMPACT executor and M1 action loss.
  M(n) classifier/planner behavior needs a separate check before use.

## First Checks To Run

1. A one-rank, one-update training smoke test with `--window-size 1`, one worker,
   a single episode, and `--profile-steps 1`; inspect that a mature sample has
   six images and all selected indices remain in the same episode.
2. Verify finite action/observation/NLL losses and gradients on COMPACT modules.
3. Run open-loop evaluation on one short episode and confirm the recorded
   summary reports v3 settings and creates a trajectory plot.
4. Compare v2 and v3 on identical held-out episodes, then validate closed-loop
   task success. A lower teacher-forced action MSE alone does not establish
   better task performance.
