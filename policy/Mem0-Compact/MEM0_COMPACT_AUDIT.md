# Mem-0 vs. Mem0-Compact Audit and Retraining Plan

**Scope.** This document records the code audit performed on September 18, 2026.
It compares the released/original `policy/Mem-0` implementation with
`policy/Mem0-Compact`, explains why a low training loss did not translate into
strong `swap_blocks` evaluation, and records the corrective changes made here.

## Executive Summary

A poor Mem0-Compact rollout does **not** yet show that COMPACT memory is
intrinsically unsuitable for RMBench. The former training recipe was not a
fair comparison with Mem-0:

1. The Compact action head ignored `repeated_diffusion_steps=8`, so it received
   roughly eight times fewer independent flow-matching targets per observation.
2. Typical Compact runs used batch size one and far fewer global samples than
   the original Mem-0 eight-GPU recipe.
3. COMPACT auxiliary losses, especially memory L2, could dominate the action
   objective.
4. Full Qwen fine-tuning with a sequential batch of one is much less stable
   than the original large-batch recipe.
5. “Best” checkpoints were selected from a stochastic training window, not a
   held-out task-relevant metric.
6. The old 30-action open-loop evaluator compounded action errors. Deployment
   now defaults to receding horizon: predict 30, execute one, observe, repeat.

The correct next experiment is an action-focused, episode-level validation
training run with a no-memory baseline alongside COMPACT.

## Architecture Comparison

### Original Mem-0

The original executor:

1. Runs Qwen3-VL normally and takes its final hidden states.
2. Pools image and text features.
3. Uses an external `MemoryBank` on image features.
4. Produces three DiT conditioning tokens: sliding-window memory, anchor
   memory, and text.
5. Stores historical memory features detached from autograd.

Relevant code:

- `policy/Mem-0/source/models/execution_module/memorymatters_executor.py:119`
- `policy/Mem-0/source/models/execution_module/memorymatters_executor.py:153`
- `policy/Mem-0/source/models/execution_module/memory_bank/memory_bank.py:440`

This is feature-memory fusion outside the Qwen decoder. There is no BPTT
through its stored history.

### Mem0-Compact

Mem0-Compact instead:

1. Adds a side-memory module to each Qwen decoder layer.
2. Predicts memory from past memory and the previous robot action/state.
3. Corrects memory from current visual hidden states.
4. Injects corrected memory back into each decoder layer.
5. Carries memory through an eight-frame TBPTT window.
6. Gives DiT only two pooled conditioning tokens: memory-modified image and
   text features.

Relevant code:

- `policy/Mem0-Compact/source/models/execution_module/mem0_compact_executor.py:302`
- `policy/Mem0-Compact/source/models/execution_module/mem0_compact_executor.py:318`
- `policy/Mem0-Compact/source/training/train_compact.py:600`

This is therefore a materially harder recurrent architecture, not a drop-in
replacement for the original MemoryBank.

## Confirmed Training Mismatches

### Diffusion repetitions

Original Mem-0 repeats actions and conditioning eight times before calling its
DiT action head. Each repeat samples independent noise and diffusion time.

- Original: `memorymatters_executor.py:153`
- Original config: `execution_module_train.yaml:27`

Before this change, Mem0-Compact had `repeated_diffusion_steps: 8` in YAML but
called the action head only once. The action head now repeats **only** the DiT
inputs/targets, after the VLM/COMPACT forward:

- `policy/Mem0-Compact/source/models/execution_module/action_model/ActionHeader.py`

This restores eight independent action/noise examples per observation without
eight expensive Qwen passes.

### Effective data exposure

The original Mem-0 config specifies batch size 56, 30,000 steps, and eight
repeats; the published recipe was run on eight A800 GPUs. A Compact run with
batch size 1, eight-frame TBPTT, 30,000 windows, and one GPU sees only about
240,000 frames. The original recipe sees approximately 13.44 million frames
before diffusion repeats, or roughly 107.5 million action/noise pairs after
repeats.

Even after restoring repeats, scaling global batch through DDP remains
important. For recurrent COMPACT, keep `BATCH_SIZE=1` **per GPU**; use `NPROC`
to add GPUs.

### Batch-size constraint

`RandomEpisodeIterableDataset` yields frames in temporal order. A DataLoader
batch larger than one assigns memory slot zero frames like `t0 -> tB -> t2B`,
not `t0 -> t1 -> t2`. That breaks the intended temporal recurrence.

Use:

```bash
BATCH_SIZE=1
```

per rank. Global batch is `NPROC`, not `BATCH_SIZE`.

## Why a Total Loss of 0.02 Is Not Enough

Compact optimizes flow action loss plus observation, NLL, memory-L2, and (for
M(n)) classifier losses. Earlier logs showed the weighted memory losses larger
than the action loss. The memory-L2 term is a sum over memory elements, so it
can encourage memory shrinkage/collapse while total loss falls.

Relevant code:

- `policy/Mem0-Compact/source/models/execution_module/mem0_compact_executor.py:343`

A low stochastic total loss can therefore mean auxiliary objectives became
easy or the model overfit the sampled window; it does not prove the predicted
action chunks work in closed-loop simulation.

## Checkpoint Selection: Implemented Change

`ckpt_best.pt` now uses a deterministic **validation action loss** when
validation is enabled:

- A fixed, seeded 10% of episodes is held out before rank sharding.
- Train and validation episode IDs are disjoint.
- Validation uses `batch_size=1`, finite sequential episode iteration, no color
  jitter, and recurrent memory reset at each episode boundary.
- Validation uses only action loss, excluding auxiliary COMPACT losses.
- Validation RNG state is restored afterwards, so it does not perturb training
  randomness.
- Under DDP, action-loss sums and frame counts are reduced across ranks.
- Checkpoints retain `best_val_action_loss` and `best_val_step`.

Relevant code:

- `policy/Mem0-Compact/source/training/train_compact.py`
- `policy/Mem0-Compact/source/dataloader/random_episode_dataloader.py`
- `policy/Mem0-Compact/source/dataloader/dataset_min_max.py`

The shipped configs validate every 1,000 windows. Change the `validation:`
block to alter the fraction, interval, seed, or to use a smaller fixed holdout
with `max_episodes`.

## Evaluation Behavior

Mem0-Compact deployment now defaults to:

```text
predict 30 actions -> execute 1 action -> observe -> predict again
```

This is receding-horizon control. It is slower than executing the full chunk,
but substantially more robust to model errors and lets side memory consume
fresh observations. Set this via `action_execute_steps: 1` in
`deploy_policy.yml`.

## Recommended Experiment Order

1. **Evaluate existing checkpoints** with `action_execute_steps=1`; compare
   `ckpt_best.pt` and `ckpt_final.pt`.
2. **Train the no-memory baseline** with the same data, training budget,
   validation split, and evaluation protocol.
3. **Freeze Qwen initially** (`FREEZE_BASE=1`) and train DiT/side-memory first.
4. **Start action-only:** set `lambda_obs=lambda_nll=lambda_mem=0` in a copied
   Compact config. Add auxiliary losses back one at a time only after action
   success is established.
5. **Use multi-GPU DDP**, with `BATCH_SIZE=1` per GPU, to restore sample
   exposure safely.
6. Select candidates using both validation action loss and periodic RMBench
   success evaluation. Validation action loss is a much better checkpoint
   filter than training loss, but simulation success is still the final metric.

## Status of Earlier Reliability Fixes

- DDP `no_sync()` covers both forward and backward for the baseline.
- Baseline gradient checkpointing uses non-reentrant checkpointing.
- All ranks jointly skip non-finite loss/gradient updates.
- Deployment advances time by actions actually executed.
- Checkpoint configuration travels with weights and reconstructs the correct
  baseline/COMPACT variant at evaluation.

These are implementation-correctness fixes; the training recommendations above
are the next experimental controls required to determine whether side memory
helps.
