"""Tests for online optical-flow visual GT."""

from __future__ import annotations

import numpy as np

from gr00t.data.visual_flow import compute_nav_flow_features, compute_waypoint_flow_features


def test_waypoint_flow_zero_at_current_frame():
    frames = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    frames[3, :, 10:20] = 255
    flow = compute_waypoint_flow_features(frames)
    assert flow.shape == (8, 2)
    assert np.allclose(flow[0], 0.0)


def test_nav_flow_averages_views():
    left = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    right = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    left[1, :, 8:] = 255
    right[1, :, :8] = 255
    nav = compute_nav_flow_features(left, right)
    assert nav.shape == (4, 2)
    assert nav[0, 0] == 0.0 and nav[0, 1] == 0.0
