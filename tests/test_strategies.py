# SPDX-License-Identifier: MulanPSL-2.0
"""The `strategy` config key, and mrtsp after Liu et al. (2025):
map optimization (map_opt.py), path-aware goals and the visiting order
(strategies.py)."""
import math

import numpy as np
import pytest

from explore_skill.frontier import (FrontierCluster, GridView,
                                    cluster_frontiers, find_frontier_cells)
from explore_skill.map_opt import optimize
from explore_skill.strategies import STRATEGIES, _mrtsp_order, config_problem, pick, validate

FREE, UNKNOWN, WALL = 0, -1, 100
RADIUS = 0.235
P = {"wd": 1.0, "ws": 1.0, "sensor_range_m": 1.5, "vmax_m_s": 0.26,
     "wmax_rad_s": 1.0, "resolution_m": 0.05}


def _grid(data):
    h, w = data.shape
    return GridView(data=data.astype(np.int8), resolution=0.05,
                    origin_x=0.0, origin_y=0.0, width=w, height=h)


def _room(w, h):
    data = np.full((h, w), FREE, dtype=np.int8)
    data[0, :] = data[-1, :] = data[:, 0] = data[:, -1] = WALL
    return data


def _mrtsp(data, robot, **kwargs):
    return pick("mrtsp", _grid(data), robot, min_size=8,
                robot_radius_m=RADIUS, **kwargs)


def test_unknown_strategies_are_rejected():
    for s in STRATEGIES:
        assert validate(s) == s
    with pytest.raises(ValueError, match="unknown explore strategy"):
        validate("tare")


def test_config_values_are_checked_at_init():
    assert config_problem({}) is None
    assert config_problem({"strategy": "mrtsp", "mrtsp": {"ws": 2, "wd": 0}}) is None
    assert "unknown explore strategy" in config_problem({"strategy": "tare"})
    assert "greater than 0" in config_problem({"mrtsp": {"ws": 0}})
    assert "at least 0" in config_problem({"mrtsp": {"sensor_range_m": -1}})
    assert "must be a number" in config_problem({"mrtsp": {"vmax_m_s": "fast"}})
    assert "not a parameter" in config_problem({"mrtsp": {"speed": 1}})


# Map optimization

def _walled_room():
    data = np.full((80, 80), UNKNOWN, dtype=np.int8)
    data[10:70, 10:70] = _room(60, 60)
    return data


def _frontiers(gv):
    return cluster_frontiers(find_frontier_cells(gv), min_size=1)


def test_sparse_beam_streaks_inside_a_room_stop_being_frontiers():
    data = _walled_room()
    for y in range(15, 65, 6):
        data[y, 20:23] = UNKNOWN
        data[y + 2, 45:46] = UNKNOWN
    assert len(_frontiers(_grid(data))) >= 10
    assert _frontiers(optimize(_grid(data))) == []


def test_a_doorway_stays_a_frontier_and_walls_are_kept():
    data = _walled_room()
    data[36:44, 69] = FREE
    opt = optimize(_grid(data))
    assert len(_frontiers(opt)) == 1
    assert (opt.data[data == WALL] == WALL).all()
    assert (opt.data[:9, :] == UNKNOWN).all()      # behind the top wall


# Goals

def test_a_frontier_behind_a_wall_is_as_far_as_the_way_round():
    data = _room(120, 60)
    data[0:46, 60] = WALL                      # gap at the top only
    data[5:16, 64:71] = UNKNOWN                # 0.85 m away through the wall
    data[40:56, 5:9] = UNKNOWN                 # 2.9 m away in the open
    target = _mrtsp(data, (2.5, 0.5))
    assert target.centroid_xy[0] < 3.0
    assert target.goal_xy[0] < 3.0             # on the robot's side


def test_a_frontier_with_no_way_to_it_is_dropped():
    data = _room(100, 60)
    data[20:41, 60:81] = WALL
    data[21:40, 61:80] = FREE
    data[25:36, 65:76] = UNKNOWN               # walled in, robot outside
    assert _mrtsp(data, (1.0, 1.0)) is None


def test_a_frontier_reached_only_under_a_table_is_dropped():
    data = _room(120, 80)
    data[0:46, 60] = WALL                      # gap at the top only
    data[5:16, 90:101] = UNKNOWN
    assert _mrtsp(data, (1.0, 0.5)) is not None
    table = [(3.0, 3.1, 0.9)]                  # fills the gap
    assert _mrtsp(data, (1.0, 0.5), keepout=table) is None


def test_the_goal_is_short_of_the_frontier_and_faces_it():
    data = _room(120, 60)
    data[1:-1, 100:-1] = UNKNOWN
    target = _mrtsp(data, (0.5, 1.5))
    (gx, gy), (fx, fy) = target.goal_xy, target.centroid_xy
    assert gx < fx - 0.4
    assert abs(math.atan2(fy - gy, fx - gx)) < math.radians(30)


def test_goals_keep_the_robot_radius_from_walls():
    data = _room(120, 60)
    data[1:-1, 100:-1] = UNKNOWN
    data[30, 20:90] = WALL                     # a wall down the middle
    gy = _mrtsp(data, (0.5, 0.5)).goal_xy[1]
    assert abs(gy - 1.5) > RADIUS and gy > 0.05 + RADIUS


# Visiting order

def _candidate(center, size):
    c = FrontierCluster(centroid_xy=center, size=size,
                        cell_indices=np.zeros((size, 2), dtype=int),
                        goal_xy=center, path_m=0.0)
    return (c, 0.0, {"dm": math.hypot(*center), "dn": math.hypot(*center),
                     "centroid": center, "entry": center})


def test_the_whole_order_decides_the_first_frontier():
    """The big frontier alone looks best, but visiting the small one first
    makes the whole order cheaper."""
    cands = [_candidate((0.0, 4.0), 10), _candidate((0.0, -4.0), 100)]
    assert _mrtsp_order(cands, (0.0, 0.0), 0.0, P) == [0, 1]


def test_a_frontier_behind_the_robot_pays_for_the_turn():
    cands = [_candidate((4.0, 0.0), 10), _candidate((-4.0, 0.0), 100)]
    assert _mrtsp_order(cands, (0.0, 0.0), 0.0, P)[0] == 0


def test_the_order_visits_every_frontier_once():
    cands = [_candidate(xy, 20)
             for xy in ((3, 0), (6, 0), (9, 0), (0, 4), (-5, -5))]
    assert sorted(_mrtsp_order(cands, (0.0, 0.0), 0.0, P)) == list(range(5))


def test_a_robot_beside_a_wall_still_finds_goals():
    """Its own cell lies in the wall's inflation; that must not wall it in.
    0.32 m is TIAGo's inscribed radius plus the margin Soma's is given."""
    data = _room(120, 60)
    data[1:-1, 100:-1] = UNKNOWN
    for gap in (0.30, 0.28, 0.26):
        target = pick("mrtsp", _grid(data), (1.5, 0.05 + gap), min_size=8,
                      robot_radius_m=0.32)
        assert target is not None, gap
