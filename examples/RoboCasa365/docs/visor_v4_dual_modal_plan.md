# VISOR v4 Dual-Modal Plan (RoboCasa365)

Phased rollout with independent checkpoints and rollback switches. Baseline: A0
(`checkpoint-30000` tactile-only VISOR T-Rex).

## Phase map

| Phase | Goal | Gate mode | Visual GT | Rollback |
|-------|------|-----------|-----------|----------|
| v4.0 | Train/infer tactile align, remove spatial_proj | `tactile_hand_only` | none | A0 |
| v4.1 | Visual readout + aux loss | `dual_split` or sup only | L1 pool | v4.0 |
| v4.2 | Triple gate: arm + base + hand | `visual_manip_nav_tactile_hand` | L1 pool | v4.1 / v4.0 |

## v4.2 action routing (flat 12-d)

| Slice | Field | Gate source |
|-------|-------|-------------|
| `[0:1]` | gripper | visual hand (pre-grasp) + tactile contact |
| `[1:7]` | EEF | visual manip (eye_in_hand waypoints) |
| `[7:11]` | base | visual nav (agentview pool waypoints) |
| `[11:12]` | control_mode | not gated |

## Data pipeline

1. Run haptic labels (if missing): `generate_haptic_gripper_labels.py`
2. Run visual labels: `scripts/run_visual_future_labels.sh`
   - Writes `visual_future.manip` / `visual_future.nav` (256D per frame)
   - Training stacks 8 waypoint deltas `[0,5,...,35]` → `(8, 256)`
3. Finetune: `finetune_pickplace_visor_v42_30k.sh`

Modality config: `robocasa365_config_4frame.py` (`visual_future` keys: manip, nav).

## Key flags (tyro / FinetuneConfig)

```bash
--visor-gate-mode visual_manip_nav_tactile_hand
--visor-use-visual-supervision
--visor-use-readout-fed-gates
--visor-visual-gt-level pool
--visor-tactile-align-mode hold_last
```

## Go / no-go gates

**v4.0 → v4.1**

- 500-step smoke: no NaN; `flow_loss` within 5% of v4.0
- Mini-eval (5k ckpt): success ≥ A0 − 3pp

**v4.1 → v4.2**

- `visual_loss` decreases in first 100 batches
- `flow_loss` Δ < 5% vs v4.1
- v4.2 mini-eval: arm motion not collapsed; base moves when needed

**v4.2 ship**

- Full 30k eval vs A0: target ≥ +5pp on PickPlaceToasterToCounter
- If arm gate ineffective: `--visor-gate-mode dual_split` (drop base gate)
- If flow degrades: revert to v4.0 + tactile_hand_only

## Files

| Component | Path |
|-----------|------|
| Core | `gr00t/model/modules/visor/visor.py` |
| Flat head | `gr00t/model/modules/visor/visor_flat_action_head.py` |
| Visual labels | `examples/RoboCasa365/scripts/generate_visual_future_labels.py` |
| v4.2 finetune | `examples/RoboCasa365/finetune_pickplace_visor_v42_30k.sh` |
