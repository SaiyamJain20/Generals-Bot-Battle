"""Reward shaping, semi-MDP discounting and GAE for RO-PPO (DESIGN.md section 4). numpy only."""
import math

import numpy as np


def potential(cnt, beta):
    """Phi(s) = beta*(0.5*tanh(log(my_army/opp_army)) + 0.5*tanh(log(my_land/opp_land)))."""
    ma, oa, ml, ol = cnt
    return beta * (0.5 * math.tanh(math.log(max(1, ma) / max(1, oa))) +
                   0.5 * math.tanh(math.log(max(1, ml) / max(1, ol))))


def shaping(turns, cnts, gamma, beta, t_end=None):
    """Per-decision shaping F_k and the elapsed turns tau_k.

    F_k = gamma^tau_k * Phi(s_{k+1}) - Phi(s_k), Phi(terminal) = 0.
    The last decision's tau = t_end - turns[-1] (game end), 1 if t_end is None; its successor is terminal."""
    n = len(turns)
    phi = [potential(c, beta) for c in cnts]
    tau_last = 1 if t_end is None else max(1, t_end - turns[-1])
    tau = [max(1, turns[k + 1] - turns[k]) for k in range(n - 1)] + [tau_last]
    F = []
    for k in range(n):
        nxt = phi[k + 1] if k + 1 < n else 0.0
        F.append(gamma ** tau[k] * nxt - phi[k])
    return np.array(F, dtype=np.float64), np.array(tau, dtype=np.float64)


def gae(rewards, values, tau, gamma, lam):
    """Semi-MDP GAE. values[k] = V(s_k); V(terminal) = 0.
    delta_k = R_k + gamma^tau_k V_{k+1} - V_k ; A_k = delta_k + gamma^tau_k lam A_{k+1}."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    nxt_a = 0.0
    for k in range(n - 1, -1, -1):
        d = gamma ** tau[k]
        v1 = values[k + 1] if k + 1 < n else 0.0
        delta = rewards[k] + d * v1 - values[k]
        nxt_a = delta + d * lam * nxt_a
        adv[k] = nxt_a
    return adv, adv + np.asarray(values, dtype=np.float64)


def episode_advantages(turns, cnts, values, outcome_reward, gamma, lam, beta, t_end=None):
    """Full pipeline for one episode: shaping + terminal reward + GAE."""
    F, tau = shaping(turns, cnts, gamma, beta, t_end)
    R = F.copy()
    # the outcome arrives on the env step just before termination: discount it back to the last decision
    gap = 0 if t_end is None else max(0, t_end - 1 - turns[-1])
    R[-1] += gamma ** gap * outcome_reward
    adv, ret = gae(R, values, tau, gamma, lam)
    return adv, ret, R, tau
