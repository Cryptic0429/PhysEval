"""Seeded synthetic trajectories shared by spring confidence validation/tests."""
from __future__ import annotations
import numpy as np


def cases():
    for period in (.4, 1., 2.5):
        for fps in (24, 30, 60):
            for cycles in (2., 4., 8.):
                yield dict(kind="clean", period=period, fps=fps, cycles=cycles, noise=0., decay=0., missing=0., seed=0)
    for seed in (11, 12, 71, 72):
        for noise in (.05, .15, .25, .5, 1.):
            for decay in (0., .15, .5):
                yield dict(kind="noisy_damped", period=1., fps=30, cycles=8., noise=noise, decay=decay, missing=0., seed=seed)
        for missing in (.1, .3, .6):
            yield dict(kind="missing", period=1., fps=30, cycles=8., noise=.05, decay=.05, missing=missing, seed=seed)
        for kind in ("linear", "quadratic", "random_walk", "white_noise", "chirp", "gap", "strong_decay"):
            yield dict(kind=kind, period=1., fps=30, cycles=8., noise=0., decay=0., missing=0., seed=seed)
        for cycles in (.6, 1., 1.4):
            yield dict(kind="short", period=1., fps=30, cycles=cycles, noise=0., decay=0., missing=0., seed=seed)


def signal(c):
    rng = np.random.default_rng(c["seed"])
    t = np.arange(0, c["cycles"] * c["period"], 1/c["fps"])
    s = np.exp(-c["decay"]*t) * np.sin(2*np.pi*t/c["period"] + .3)
    if c["kind"] == "linear": s = t
    if c["kind"] == "quadratic": s = t*t
    if c["kind"] == "random_walk": s = np.cumsum(rng.normal(size=len(t)))
    if c["kind"] == "white_noise": s = rng.normal(size=len(t))
    if c["kind"] == "chirp": s = np.sin(2*np.pi*(t+.12*t*t))
    if c["kind"] == "strong_decay": s = np.exp(-2*t)*np.sin(2*np.pi*t+.3)
    s = s + rng.normal(0, c["noise"], len(t))
    keep = rng.random(len(t)) >= c["missing"]
    if c["kind"] == "gap": keep &= ~((t>2) & (t<4))
    return t[keep], s[keep]
