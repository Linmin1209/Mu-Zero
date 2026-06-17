#!/usr/bin/env python3
"""DynaMem-style decomposed controller on RoboCasa365 sim (Panda-Omron).

Pipeline: navigate -> pick -> navigate -> place.

Manip modes (env ``DYNAMEM_MANIP_MODE``):
  - ``oracle`` (default): after nav, snap pick/place in sim (decomposition baseline)
  - ``osc``: proportional delta OSC on the Panda arm (slow / fragile on PnP)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from robosuite.utils import transform_utils as T
from tqdm import tqdm

import robocasa  # noqa: F401
from robocasa.utils import env_utils as EnvUtils
from robocasa.utils import object_utils as OU

ManipMode = Literal["oracle", "osc"]
DEFAULT_MANIP_MODE: ManipMode = os.environ.get("DYNAMEM_MANIP_MODE", "oracle")  # type: ignore[assignment]


class Phase(Enum):
    NAV_TO_SRC = auto()
    PICK = auto()
    NAV_TO_DST = auto()
    PLACE = auto()
    NAV_ONLY = auto()
    DONE = auto()


class ManipSubPhase(Enum):
    HOVER = auto()
    APPROACH = auto()
    ACT = auto()  # grasp or release
    RETREAT = auto()


@dataclass
class OvmmTaskConfig:
    obj_name: str
    src_attr: str
    dst_attr: str
    place_receptacle: str | None = None
    place_mode: str = "receptacle"  # receptacle | inside_fixture | counter_surface
    pick_hover_z: float = 0.16
    pick_approach_z: float = 0.045
    pick_lift_z: float = 0.18
    place_hover_z: float = 0.14
    place_approach_z: float = 0.06


OVMM_TASK_CONFIGS: dict[str, OvmmTaskConfig] = {
    "PickPlaceCounterToCabinet": OvmmTaskConfig(
        "obj", "counter", "cab", place_mode="inside_fixture", pick_hover_z=0.14
    ),
    "PickPlaceCounterToStove": OvmmTaskConfig(
        "obj",
        "counter",
        "stove",
        place_receptacle="container",
        pick_hover_z=0.14,
    ),
    "PickPlaceDrawerToCounter": OvmmTaskConfig(
        "obj",
        "drawer",
        "counter",
        place_mode="counter_surface",
        pick_hover_z=0.12,
        pick_approach_z=0.03,
    ),
    "PickPlaceSinkToCounter": OvmmTaskConfig(
        "obj",
        "sink",
        "counter",
        place_receptacle="container",
        pick_hover_z=0.14,
        pick_approach_z=0.05,
    ),
    "PickPlaceToasterToCounter": OvmmTaskConfig(
        "obj",
        "toaster",
        "counter",
        place_receptacle="plate",
        pick_hover_z=0.15,
        pick_approach_z=0.04,
    ),
    "DeliverStraw": OvmmTaskConfig(
        "straw",
        "drawer",
        "dining_counter",
        place_receptacle="glass_cup",
        pick_hover_z=0.12,
        pick_approach_z=0.025,
        place_approach_z=0.05,
    ),
}


def unwrap_robocasa_env(gym_env: gym.Env):
    env = gym_env
    while hasattr(env, "env"):
        env = env.env
    return env


def _zero_action() -> dict[str, np.ndarray]:
    return {
        "action.end_effector_position": np.zeros(3, dtype=np.float32),
        "action.end_effector_rotation": np.zeros(3, dtype=np.float32),
        "action.gripper_close": np.array([0.0], dtype=np.float32),
        "action.base_motion": np.zeros(4, dtype=np.float32),
        "action.control_mode": np.array([0.0], dtype=np.float32),
    }


def _arm_base_pose_from_obs(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    base_pos = np.asarray(obs["state.base_position"], dtype=float)
    base_quat = np.asarray(obs["state.base_rotation"], dtype=float)
    base_ori = T.mat2euler(T.quat2mat(T.convert_quat(base_quat, to="xyzw")))
    return base_pos, base_ori


def _base_pose(env) -> tuple[np.ndarray, np.ndarray]:
    robot_id = env.sim.model.body_name2id("mobilebase0_base")
    base_pos = np.array(env.sim.data.body_xpos[robot_id], dtype=float)
    base_ori = T.mat2euler(np.array(env.sim.data.body_xmat[robot_id]).reshape(3, 3))
    return base_pos, base_ori


def _world_to_arm_base(obs: dict[str, Any], env, world_pos: np.ndarray) -> np.ndarray:
    # Matches robot0_base_to_eef_pos frame: offset from robot0_base_pos (includes z=0.7).
    base_pos = np.asarray(obs["state.base_position"], dtype=float)
    return world_pos - base_pos


def _obj_world_pos(env, obj_name: str) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]], dtype=float)


def _world_to_base(obs: dict[str, Any], env, world_pos: np.ndarray) -> np.ndarray:
    return _world_to_arm_base(obs, env, world_pos)


def _eef_rel_pos(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["state.end_effector_position_relative"], dtype=float)


def _eef_rel_quat(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs["state.end_effector_rotation_relative"], dtype=float)


def _nav_target_reached(env, target_pos: np.ndarray, target_ori: np.ndarray) -> bool:
    base_pos, base_ori = _base_pose(env)
    pos_ok = np.linalg.norm(target_pos[:2] - base_pos[:2]) <= 0.20
    ori_ok = np.cos(target_ori[2] - base_ori[2]) >= 0.98
    return bool(pos_ok and ori_ok)


def _base_nav_action(
    env,
    target_pos: np.ndarray,
    target_ori: np.ndarray,
    pos_scale: float = 0.10,
    rot_scale: float = 0.25,
) -> dict[str, np.ndarray]:
    base_pos, base_ori = _base_pose(env)
    dxy_world = target_pos[:2] - base_pos[:2]
    dist = np.linalg.norm(dxy_world)
    yaw = base_ori[2]
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    dx = cos_y * dxy_world[0] + sin_y * dxy_world[1]
    dy = -sin_y * dxy_world[0] + cos_y * dxy_world[1]

    desired_yaw = np.arctan2(dxy_world[1], dxy_world[0]) if dist > 1e-3 else target_ori[2]
    dtheta = desired_yaw - yaw
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    final_yaw_err = target_ori[2] - yaw
    final_yaw_err = (final_yaw_err + np.pi) % (2 * np.pi) - np.pi

    action = _zero_action()
    action["action.control_mode"] = np.array([1.0], dtype=np.float32)
    if dist > 0.35 and abs(dtheta) > 0.25:
        move = np.array([0.0, 0.0, np.clip(dtheta / rot_scale, -1.0, 1.0), 0.0], dtype=np.float32)
    elif dist > 0.22:
        move = np.array(
            [
                np.clip(dx / pos_scale, -1.0, 1.0),
                np.clip(dy / pos_scale, -1.0, 1.0),
                np.clip(0.5 * dtheta / rot_scale, -1.0, 1.0),
                0.0,
            ],
            dtype=np.float32,
        )
    else:
        move = np.array(
            [
                np.clip(dx / max(pos_scale * 0.5, 0.05), -1.0, 1.0),
                np.clip(dy / max(pos_scale * 0.5, 0.05), -1.0, 1.0),
                np.clip(final_yaw_err / rot_scale, -1.0, 1.0),
                0.0,
            ],
            dtype=np.float32,
        )
    action["action.base_motion"] = move
    return action


def _arm_action_toward(
    obs: dict[str, Any],
    target_rel: np.ndarray,
    gripper_close: float,
    gain: float = 8.0,
) -> dict[str, np.ndarray]:
    current = _eef_rel_pos(obs)
    delta = target_rel - current

    action = _zero_action()
    action["action.control_mode"] = np.array([0.0], dtype=np.float32)
    action["action.end_effector_position"] = np.clip(
        gain * delta, -1.0, 1.0
    ).astype(np.float32)
    # Nudge gripper to point downward (tool z aligned with -world z in base frame).
    quat = _eef_rel_quat(obs)
    if quat.shape[0] == 4:
        rot_mat = T.quat2mat(T.convert_quat(quat, to="xyzw"))
        tool_z = rot_mat[:, 2]
        desired_z = np.array([0.0, 0.0, -1.0])
        rot_err = np.cross(tool_z, desired_z)
        action["action.end_effector_rotation"] = np.clip(
            0.8 * rot_err, -1.0, 1.0
        ).astype(np.float32)
    action["action.gripper_close"] = np.array([gripper_close], dtype=np.float32)
    return action


def _obj_rel_pos(obs: dict[str, Any], env, obj_name: str, z_offset: float = 0.0) -> np.ndarray:
    world = _obj_world_pos(env, obj_name) + np.array([0.0, 0.0, z_offset])
    return _world_to_base(obs, env, world)


def _placement_pose(obs: dict[str, Any], env, cfg: OvmmTaskConfig) -> np.ndarray:
    if cfg.place_receptacle is not None and cfg.place_mode == "receptacle":
        world = _obj_world_pos(env, cfg.place_receptacle)
        return _world_to_base(obs, env, world + np.array([0.0, 0.0, cfg.place_approach_z]))

    if cfg.place_mode == "inside_fixture":
        dst = getattr(env, cfg.dst_attr)
        world = np.array(dst.pos, dtype=float) + np.array([0.0, 0.0, 0.08])
        return _world_to_base(obs, env, world)

    if cfg.place_mode == "counter_surface":
        dst = getattr(env, cfg.dst_attr)
        world = np.array(dst.pos, dtype=float) + np.array([0.0, 0.0, 0.12])
        return _world_to_base(obs, env, world)

    dst = getattr(env, cfg.dst_attr)
    world = np.array(dst.pos, dtype=float) + np.array([0.0, 0.0, 0.12])
    return _world_to_base(obs, env, world)


def _prepare_env(env, task_name: str) -> None:
    if hasattr(env, "cab") and hasattr(env.cab, "open_door"):
        env.cab.open_door(env=env)
    if hasattr(env, "drawer") and hasattr(env.drawer, "open_door"):
        env.drawer.open_door(env, min=1.0, max=1.0)
    if task_name == "DeliverStraw" and hasattr(env, "drawer"):
        env.drawer.open_door(env, min=1.0, max=1.0)


def _is_grasped(env, obj_name: str) -> bool:
    try:
        return bool(OU.check_obj_grasped(env, obj_name))
    except Exception:
        return False


def _eef_world_pos(env) -> np.ndarray:
    site_id = env.robots[0].eef_site_id["right"]
    return np.array(env.sim.data.site_xpos[site_id], dtype=float)


def _set_object_world_pose(env, obj_name: str, pos: np.ndarray) -> None:
    obj = env.objects[obj_name]
    body_id = env.obj_body_id[obj_name]
    quat = np.array(env.sim.data.body_xquat[body_id], dtype=float)
    qpos = np.concatenate([pos, quat])
    with EnvUtils.no_collision(env.sim):
        env.sim.data.set_joint_qpos(obj.joints[0], qpos)
        env.sim.forward()


def _oracle_place_world(env, cfg: OvmmTaskConfig) -> np.ndarray:
    if cfg.place_receptacle is not None:
        return _obj_world_pos(env, cfg.place_receptacle) + np.array([0.0, 0.0, 0.03])
    if cfg.place_mode == "inside_fixture":
        dst = getattr(env, cfg.dst_attr)
        return np.array(dst.pos, dtype=float) + np.array([0.0, 0.0, 0.06])
    if cfg.place_mode == "counter_surface":
        dst = getattr(env, cfg.dst_attr)
        return np.array(dst.pos, dtype=float) + np.array([0.0, 0.0, 0.12])
    dst = getattr(env, cfg.dst_attr)
    return np.array(dst.pos, dtype=float) + np.array([0.0, 0.0, 0.12])


def _oracle_grasp(env, obj_name: str) -> None:
    eef = _eef_world_pos(env)
    _set_object_world_pose(env, obj_name, eef + np.array([0.0, 0.0, -0.02]))


def _oracle_weld_to_eef(env, obj_name: str) -> None:
    _oracle_grasp(env, obj_name)


def _oracle_release(env, cfg: OvmmTaskConfig) -> None:
    _set_object_world_pose(env, cfg.obj_name, _oracle_place_world(env, cfg))


class DynaMemRoboCasaSimController:
    """State machine: navigate / pick / place decomposition."""

    def __init__(
        self,
        task_name: str,
        compat_status: str,
        max_episode_steps: int = 720,
        phase_step_budget: int = 300,
        manip_mode: ManipMode | None = None,
    ):
        self.task_name = task_name
        self.compat_status = compat_status
        self.max_episode_steps = max_episode_steps
        self.phase_step_budget = phase_step_budget
        self.manip_mode: ManipMode = manip_mode or DEFAULT_MANIP_MODE
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.DONE
        self.manip_sub = ManipSubPhase.HOVER
        self.step_in_phase = 0
        self.gripper_closed = False
        self.object_welded = False
        self.nav_target_pos: np.ndarray | None = None
        self.nav_target_ori: np.ndarray | None = None
        self.ovmm_cfg: OvmmTaskConfig | None = None

        if self.compat_status == "navigate_only":
            self.phase = Phase.NAV_ONLY
        elif self.compat_status == "ovmm":
            self.ovmm_cfg = OVMM_TASK_CONFIGS.get(self.task_name)
            if self.ovmm_cfg is not None:
                self.phase = Phase.NAV_TO_SRC

    def _advance_phase(self) -> None:
        self.step_in_phase = 0
        self.manip_sub = ManipSubPhase.HOVER
        if self.phase == Phase.NAV_TO_SRC:
            self.phase = Phase.PICK
        elif self.phase == Phase.PICK:
            self.phase = Phase.NAV_TO_DST
        elif self.phase == Phase.NAV_TO_DST:
            self.phase = Phase.PLACE
        elif self.phase == Phase.PLACE:
            self.phase = Phase.DONE

    def _advance_manip_sub(self) -> None:
        self.step_in_phase = 0
        if self.manip_sub == ManipSubPhase.HOVER:
            self.manip_sub = ManipSubPhase.APPROACH
        elif self.manip_sub == ManipSubPhase.APPROACH:
            self.manip_sub = ManipSubPhase.ACT
        elif self.manip_sub == ManipSubPhase.ACT:
            self.manip_sub = ManipSubPhase.RETREAT
        elif self.manip_sub == ManipSubPhase.RETREAT:
            self.manip_sub = ManipSubPhase.RETREAT

    def _set_nav_target(self, env, fixture) -> None:
        self.nav_target_pos, self.nav_target_ori = EnvUtils.compute_robot_base_placement_pose(
            env, fixture
        )

    def on_reset(self, env) -> None:
        _prepare_env(env, self.task_name)
        if self.phase == Phase.NAV_ONLY and hasattr(env, "target_fixture"):
            self._set_nav_target(env, env.target_fixture)
        elif self.phase == Phase.NAV_TO_SRC and self.ovmm_cfg is not None:
            src = getattr(env, self.ovmm_cfg.src_attr)
            self._set_nav_target(env, src)

    def _pick_target(self, obs: dict[str, Any], env, cfg: OvmmTaskConfig) -> np.ndarray:
        z_map = {
            ManipSubPhase.HOVER: cfg.pick_hover_z,
            ManipSubPhase.APPROACH: cfg.pick_approach_z,
            ManipSubPhase.ACT: cfg.pick_approach_z,
            ManipSubPhase.RETREAT: cfg.pick_lift_z,
        }
        return _obj_rel_pos(obs, env, cfg.obj_name, z_map[self.manip_sub])

    def _place_target(self, obs: dict[str, Any], env, cfg: OvmmTaskConfig) -> np.ndarray:
        base = _placement_pose(obs, env, cfg)
        if self.manip_sub == ManipSubPhase.HOVER:
            base[2] += cfg.place_hover_z - cfg.place_approach_z
        elif self.manip_sub == ManipSubPhase.RETREAT:
            base[2] += cfg.place_hover_z
        return base

    def _run_pick_oracle(self, obs: dict[str, Any], env) -> dict[str, np.ndarray]:
        assert self.ovmm_cfg is not None
        cfg = self.ovmm_cfg
        if not self.gripper_closed:
            if self.step_in_phase <= 8:
                target = _obj_rel_pos(obs, env, cfg.obj_name, cfg.pick_hover_z)
                return _arm_action_toward(obs, target, gripper_close=0.0, gain=8.0)
            _oracle_grasp(env, cfg.obj_name)
            self.gripper_closed = True
            self.object_welded = True
            self._advance_phase()
            dst = getattr(env, cfg.dst_attr)
            self._set_nav_target(env, dst)
            return _arm_action_toward(obs, _eef_rel_pos(obs), gripper_close=1.0, gain=1.0)
        return _zero_action()

    def _run_place_oracle(self, obs: dict[str, Any], env) -> dict[str, np.ndarray]:
        assert self.ovmm_cfg is not None
        cfg = self.ovmm_cfg
        if self.gripper_closed:
            if self.step_in_phase <= 5:
                target = _world_to_arm_base(
                    obs, env, _oracle_place_world(env, cfg) + np.array([0.0, 0.0, 0.08])
                )
                return _arm_action_toward(obs, target, gripper_close=1.0, gain=8.0)
            _oracle_release(env, cfg)
            self.gripper_closed = False
            self.object_welded = False
            self._advance_phase()
            return _arm_action_toward(obs, _eef_rel_pos(obs), gripper_close=0.0, gain=1.0)
        if self.step_in_phase > 10:
            self._advance_phase()
        return _zero_action()

    def _run_pick(self, obs: dict[str, Any], env) -> dict[str, np.ndarray]:
        if self.manip_mode == "oracle":
            return self._run_pick_oracle(obs, env)
        assert self.ovmm_cfg is not None
        cfg = self.ovmm_cfg
        target = self._pick_target(obs, env, cfg)
        dist = np.linalg.norm(_eef_rel_pos(obs) - target)

        if self.manip_sub == ManipSubPhase.HOVER and dist < 0.08:
            self._advance_manip_sub()
            target = self._pick_target(obs, env, cfg)
            dist = np.linalg.norm(_eef_rel_pos(obs) - target)

        if self.manip_sub == ManipSubPhase.APPROACH and dist < 0.03:
            self._advance_manip_sub()

        if self.manip_sub == ManipSubPhase.ACT:
            if not self.gripper_closed:
                if dist < 0.04:
                    self.gripper_closed = True
            elif _is_grasped(env, cfg.obj_name) or self.step_in_phase > 40:
                self._advance_manip_sub()
                target = self._pick_target(obs, env, cfg)

        if self.manip_sub == ManipSubPhase.RETREAT:
            if _is_grasped(env, cfg.obj_name) and dist < 0.05:
                self._advance_phase()
                dst = getattr(env, cfg.dst_attr)
                self._set_nav_target(env, dst)
            elif self.step_in_phase > self.phase_step_budget:
                # Failed grasp; retry pick from hover once.
                self.manip_sub = ManipSubPhase.HOVER
                self.gripper_closed = False
                self.step_in_phase = 0
                target = self._pick_target(obs, env, cfg)

        grip = 1.0 if self.gripper_closed else 0.0
        return _arm_action_toward(obs, target, gripper_close=grip, gain=10.0)

    def _run_place(self, obs: dict[str, Any], env) -> dict[str, np.ndarray]:
        if self.manip_mode == "oracle":
            return self._run_place_oracle(obs, env)
        assert self.ovmm_cfg is not None
        cfg = self.ovmm_cfg
        target = self._place_target(obs, env, cfg)
        dist = np.linalg.norm(_eef_rel_pos(obs) - target)

        if self.manip_sub == ManipSubPhase.HOVER and dist < 0.05:
            self._advance_manip_sub()
            target = self._place_target(obs, env, cfg)
            dist = np.linalg.norm(_eef_rel_pos(obs) - target)

        if self.manip_sub == ManipSubPhase.APPROACH and dist < 0.04:
            self._advance_manip_sub()

        if self.manip_sub == ManipSubPhase.ACT:
            if self.gripper_closed and dist < 0.05:
                self.gripper_closed = False
            elif not self.gripper_closed and self.step_in_phase > 25:
                self._advance_manip_sub()
                target = self._place_target(obs, env, cfg)

        if self.manip_sub == ManipSubPhase.RETREAT and dist < 0.06:
            self._advance_phase()

        grip = 1.0 if self.gripper_closed else 0.0
        return _arm_action_toward(obs, target, gripper_close=grip, gain=8.0)

    def get_action(self, obs: dict[str, Any], env) -> dict[str, np.ndarray]:
        if self.phase == Phase.DONE:
            return _zero_action()

        if env._check_success():
            return _zero_action()

        self.step_in_phase += 1

        if self.phase in (Phase.NAV_ONLY, Phase.NAV_TO_SRC, Phase.NAV_TO_DST):
            if self.nav_target_pos is None or self.nav_target_ori is None:
                return _zero_action()
            action = None
            if _nav_target_reached(env, self.nav_target_pos, self.nav_target_ori):
                if self.phase == Phase.NAV_ONLY:
                    return _zero_action()
                self._advance_phase()
                if self.phase == Phase.NAV_TO_DST and self.ovmm_cfg is not None:
                    dst = getattr(env, self.ovmm_cfg.dst_attr)
                    self._set_nav_target(env, dst)
            else:
                action = _base_nav_action(env, self.nav_target_pos, self.nav_target_ori)
            if action is None and self.phase in (
                Phase.NAV_ONLY,
                Phase.NAV_TO_SRC,
                Phase.NAV_TO_DST,
            ):
                action = _base_nav_action(env, self.nav_target_pos, self.nav_target_ori)
            if action is not None:
                if (
                    self.object_welded
                    and self.ovmm_cfg is not None
                    and self.phase == Phase.NAV_TO_DST
                ):
                    _oracle_weld_to_eef(env, self.ovmm_cfg.obj_name)
                return action

        if self.phase == Phase.PICK and self.ovmm_cfg is not None:
            return self._run_pick(obs, env)

        if self.phase == Phase.PLACE and self.ovmm_cfg is not None:
            return self._run_place(obs, env)

        return _zero_action()


def run_dynamem_sim_rollouts(
    task_name: str,
    compat_status: str,
    split: str = "pretrain",
    n_episodes: int = 50,
    max_episode_steps: int = 720,
    seed: int | None = 0,
    manip_mode: ManipMode | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    env_id = f"robocasa/{task_name}"
    make_kwargs: dict[str, Any] = {"split": split, "enable_render": False}
    if seed is not None:
        make_kwargs["seed"] = seed
    env = gym.make(env_id, **make_kwargs)

    successes: list[bool] = []
    episode_lengths: list[int] = []

    for ep in tqdm(range(n_episodes), desc=f"{task_name}", leave=False):
        ep_seed = None if seed is None else int(seed) + ep
        if ep_seed is not None:
            obs, _ = env.reset(seed=ep_seed)
        else:
            obs, _ = env.reset()

        controller = DynaMemRoboCasaSimController(
            task_name=task_name,
            compat_status=compat_status,
            max_episode_steps=max_episode_steps,
            manip_mode=manip_mode,
        )
        inner = unwrap_robocasa_env(env)
        controller.on_reset(inner)

        success = False
        for step_i in range(max_episode_steps):
            action = controller.get_action(obs, inner)
            obs, _reward, terminated, truncated, info = env.step(action)
            if info.get("success") or inner._check_success():
                success = True
                episode_lengths.append(step_i + 1)
                break
            if terminated or truncated:
                episode_lengths.append(step_i + 1)
                break
        else:
            episode_lengths.append(max_episode_steps)
        successes.append(success)

    env.close()
    rate = float(np.mean(successes)) if successes else 0.0
    return {
        "task": task_name,
        "n_episodes": n_episodes,
        "success_rate": rate,
        "successes": successes,
        "episode_lengths": episode_lengths,
        "manip_mode": manip_mode or DEFAULT_MANIP_MODE,
    }
