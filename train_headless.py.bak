"""
GUI'siz (headless) egitim.

Proje kokunden calistir (ground_main.py ile ayni klasor):
    python train_headless.py
    python train_headless.py --minutes 60
    python train_headless.py --fresh --episodes 30000
    python train_headless.py --zones --zone-count 2 4 --zone-radius 60 140

--zones ile yavaslatma bolgeli model egitilir (ayri kayit:
./model/drone_dqn_zones_*.pth). Kendi kaydi yoksa normal modelden baslar.
Bolgeler her bolumde seed'li ve rastgele uretilir; ajana verilmez.

- Ayni SimulationEngine ve ayni DroneDQNAlgorithm kullanilir, sadece cizim yok.
- Mufredat: kolay senaryolardan (yakin hedef, siki dizilim) GUI'deki
  uretecilere kadar. Asama, son 100 egitim bolumundeki basari orani
  esigi gecince ilerler. Bunlar kural degil, senaryo ayaridir.
- Belirli araliklarla seed'li sabit senaryolarda epsilon=0 degerlendirme.
  GUI'deki dagilima (son asama) gore en iyi model
  ./model/drone_dqn_shared_best.pth olarak kaydedilir. GUI'de Start
  (egitim kapali) bu dosyayi yukler.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from algorithms.model import DroneDQNAlgorithm, DroneDQNSlowZoneAlgorithm
from algorithms.reinforcementagent import MAX_EPISODE_STEPS, SharedAgent
from simulation_core.config import SimulationConfig
from simulation_core.engine import SimulationEngine
from simulation_core.slow_zones import SlowZoneConfig


CANVAS_W = 960
CANVAS_H = 760
BORDER = 40
MIN_GOAL_SEP = 50.0

PROMOTE_WINDOW = 100
PROMOTE_SUCCESS = 0.6
MIN_STAGE_EPISODES = 200


@dataclass(frozen=True)
class Stage:
    name: str
    uavs: Tuple[int, int]
    goals: Tuple[int, int]
    comm_factor: Tuple[float, float]  # drone'lar arasi yerlesim, x comm_thr
    goal_dist: Tuple[float, float]    # hedefin drone'a uzakligi, x comm_thr
    ground_prob: float
    use_engine_generators: bool = False


STAGES: Tuple[Stage, ...] = (
    Stage("s1_kolay", (2, 3), (2, 3), (0.5, 0.6), (0.5, 1.2), 0.0),
    Stage("s2", (3, 4), (3, 5), (0.6, 0.7), (0.5, 1.8), 0.3),
    Stage("s3", (3, 5), (5, 8), (0.7, 0.8), (0.5, 2.5), 0.3),
    Stage("s4", (3, 6), (8, 11), (0.8, 0.9), (0.5, 3.0), 0.3),
    Stage("gui", (3, 6), (8, 15), (0.9, 0.9), (0.0, 0.0), 0.3, True),
)


# ====================================================================== #
# Senaryo                                                                #
# ====================================================================== #

def _inside(x: float, y: float, w: int, h: int) -> bool:
    return BORDER <= x <= w - BORDER and BORDER <= y <= h - BORDER


def _far_enough(x: float, y: float, avoid: Sequence[Tuple[float, float]], sep: float) -> bool:
    return all(math.hypot(x - ax, y - ay) >= sep for ax, ay in avoid)


def _sample_near(
    rng: random.Random,
    anchors: Sequence[Tuple[float, float]],
    rmin: float,
    rmax: float,
    w: int,
    h: int,
    avoid: Sequence[Tuple[float, float]] = (),
    sep: float = 0.0,
) -> Tuple[float, float]:
    for _ in range(400):
        ax, ay = rng.choice(anchors)
        ang = rng.uniform(0.0, 2.0 * math.pi)
        r = rng.uniform(rmin, rmax)
        x, y = ax + r * math.cos(ang), ay + r * math.sin(ang)
        if _inside(x, y, w, h) and _far_enough(x, y, avoid, sep):
            return x, y
    for _ in range(400):
        x = rng.uniform(BORDER, w - BORDER)
        y = rng.uniform(BORDER, h - BORDER)
        if _far_enough(x, y, avoid, sep):
            return x, y
    return rng.uniform(BORDER, w - BORDER), rng.uniform(BORDER, h - BORDER)


def build_scenario(engine: SimulationEngine, stage: Stage, seed: int, w: int, h: int) -> None:
    rng = random.Random(seed)
    env = engine.environment
    env.reset()

    n_uav = rng.randint(*stage.uavs)
    n_goal = rng.randint(*stage.goals)
    use_ground = rng.random() < stage.ground_prob
    cx = rng.uniform(0.3 * w, 0.7 * w)
    cy = rng.uniform(0.3 * h, 0.7 * h)

    if use_ground:
        engine.generate_ground(int(round(cx)), int(round(cy)))

    # Yavaslatma bolgeleri (config yoksa ya da kapaliysa bos kalir).
    engine.generate_slow_zones(seed=seed + 7_777_777)

    if stage.use_engine_generators:
        # GUI ile ayni ureteciler (global random kullaniyorlar).
        random.seed(seed)
        engine.generate_goals(n_goal, w, h)
        engine.generate_uavs(n_uav, w, h)
        return

    thr = env.comm_thr
    factor = rng.uniform(*stage.comm_factor)
    nodes: List[Tuple[float, float]] = [(cx, cy)] if use_ground else []
    uav_pts: List[Tuple[float, float]] = []
    for _ in range(n_uav):
        if nodes:
            p = _sample_near(rng, nodes, 0.3 * thr, factor * thr, w, h)
        else:
            p = (cx, cy)
        engine.add_uav(int(round(p[0])), int(round(p[1])))
        nodes.append(p)
        uav_pts.append(p)

    goal_pts: List[Tuple[float, float]] = []
    for _ in range(n_goal):
        p = _sample_near(
            rng,
            uav_pts,
            stage.goal_dist[0] * thr,
            stage.goal_dist[1] * thr,
            w,
            h,
            goal_pts,
            MIN_GOAL_SEP,
        )
        engine.add_goal(int(round(p[0])), int(round(p[1])))
        goal_pts.append(p)


# ====================================================================== #
# Bolum ve degerlendirme                                                 #
# ====================================================================== #

def run_episode(engine: SimulationEngine, alg: DroneDQNAlgorithm, max_steps: int) -> dict:
    engine.algorithm = alg
    alg.last_episode = None
    engine.start_simulation()
    stop = None
    for _ in range(max_steps + 50):
        stop = engine.move_uavs()
        if stop is not None:
            break
    engine.stop_simulation()

    if alg.last_episode is not None:
        return alg.last_episode
    goals = engine.environment.goals
    return {
        "reward": float("nan"),
        "visited": sum(1 for g in goals if g.state == "Visited"),
        "goals": len(goals),
        "steps": max_steps,
        "end": stop or "hard_limit",
        "role_changes": 0,
        "rounds": 0,
    }


def summarize(infos: Sequence[dict]) -> Dict[str, float]:
    if not infos:
        return {"success": 0.0, "visited": 0.0, "disconnect": 0.0, "reward": 0.0, "steps": 0.0, "score": 0.0}
    success = float(np.mean([i["end"] == "all_goals_visited" for i in infos]))
    visited = float(np.mean([i["visited"] / max(i["goals"], 1) for i in infos]))
    disconnect = float(np.mean([i["end"] == "network_disconnected" for i in infos]))
    rewards = [i["reward"] for i in infos if not math.isnan(i["reward"])]
    return {
        "success": success,
        "visited": visited,
        "disconnect": disconnect,
        "reward": float(np.mean(rewards)) if rewards else float("nan"),
        "steps": float(np.mean([i["steps"] for i in infos])),
        "score": success + visited,
    }


def evaluate(
    engine: SimulationEngine,
    alg: DroneDQNAlgorithm,
    stage: Stage,
    seeds: Sequence[int],
    w: int,
    h: int,
    max_steps: int,
) -> Dict[str, float]:
    infos = []
    for s in seeds:
        build_scenario(engine, stage, s, w, h)
        infos.append(run_episode(engine, alg, max_steps))
    return summarize(infos)


def _fmt(stats: Dict[str, float]) -> str:
    return (
        f"basari {stats['success']:.2f} | ziyaret {stats['visited']:.2f} | "
        f"kopma {stats['disconnect']:.2f} | odul {stats['reward']:.2f} | "
        f"adim {stats['steps']:.0f}"
    )


# ====================================================================== #
# Ana dongu                                                              #
# ====================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(description="Drone DQN headless egitim")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--minutes", type=float, default=0.0, help="0 = sure siniri yok")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--comm-thr", type=float, default=200.0)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=CANVAS_W)
    parser.add_argument("--height", type=int, default=CANVAS_H)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--fresh", action="store_true", help="kayitli modeli yukleme, sifirdan basla")
    parser.add_argument("--zones", action="store_true", help="yavaslatma bolgeli modeli egit")
    parser.add_argument("--zone-count", type=int, nargs=2, default=(2, 4), metavar=("MIN", "MAX"))
    parser.add_argument("--zone-radius", type=float, nargs=2, default=(60.0, 140.0), metavar=("MIN", "MAX"))
    parser.add_argument("--zone-factor", type=float, default=0.5, help="bolge icinde hiz carpani")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    random.seed(args.seed)
    np.random.seed(args.seed)

    engine = SimulationEngine(comm_thr=args.comm_thr)
    engine.set_config_provider(
        lambda: SimulationConfig(target_eval_mode="single_visit", uav_speed=args.speed)
    )

    alg_cls = DroneDQNSlowZoneAlgorithm if args.zones else DroneDQNAlgorithm
    checkpoint_last = alg_cls.checkpoint_last
    checkpoint_best = alg_cls.checkpoint_best

    if args.zones:
        engine.set_slow_zone_config(
            SlowZoneConfig(
                enabled=True,
                count_min=args.zone_count[0],
                count_max=args.zone_count[1],
                radius_min=args.zone_radius[0],
                radius_max=args.zone_radius[1],
                speed_factor=args.zone_factor,
                canvas_width=args.width,
                canvas_height=args.height,
                regenerate_on_start=False,
            )
        )
        print(
            f"Yavaslatma bolgeleri: {args.zone_count[0]}-{args.zone_count[1]} adet, "
            f"yaricap {args.zone_radius[0]:.0f}-{args.zone_radius[1]:.0f}, carpan {args.zone_factor}"
        )

    agent = SharedAgent(seed=args.seed)
    if not args.fresh:
        loaded = alg_cls.load_into(agent, training=True)
        if loaded is not None:
            print(f"Yuklendi: {loaded} (karar {agent.decisions}, guncelleme {agent.updates})")
        else:
            print("Kayit yok, sifirdan basliyor.")
    else:
        agent.best_avg = float("-inf")

    train_alg = alg_cls(
        max_episode_steps=args.max_steps, seed=args.seed, verbose=False, auto_save=False
    )
    train_alg.enable_training(True)
    train_alg.agent = agent

    eval_alg = alg_cls(
        max_episode_steps=args.max_steps, seed=args.seed, verbose=False, auto_save=False
    )
    eval_alg.enable_training(False)
    eval_alg.agent = agent

    stage_idx = max(0, min(args.start_stage, len(STAGES) - 1))
    stage_hist: deque = deque(maxlen=PROMOTE_WINDOW)
    stage_episodes = 0
    recent: deque = deque(maxlen=args.log_every)
    eval_seeds = [10_000_000 + k for k in range(args.eval_episodes)]
    seed_base = args.seed * 1_000_003

    print(f"Model: {alg_cls.display_name} | Asama: {STAGES[stage_idx].name}")
    t0 = time.time()
    t_log = t0
    ep = 0
    try:
        while ep < args.episodes:
            if args.minutes > 0 and time.time() - t0 >= args.minutes * 60:
                print("Sure doldu.")
                break

            stage = STAGES[stage_idx]
            build_scenario(engine, stage, seed_base + ep, args.width, args.height)
            info = run_episode(engine, train_alg, args.max_steps)
            ep += 1
            stage_episodes += 1
            stage_hist.append(info["end"] == "all_goals_visited")
            recent.append(info)

            if ep % args.log_every == 0:
                now = time.time()
                rate = args.log_every / max(now - t_log, 1e-9)
                t_log = now
                loss = f"{agent.last_loss:.4f}" if agent.last_loss is not None else "n/a"
                print(
                    f"[ep {ep}] {stage.name} | {_fmt(summarize(list(recent)))} | "
                    f"eps {agent.epsilon:.3f} | guncelleme {agent.updates} | "
                    f"loss {loss} | replay {len(agent.memory)} | {rate:.1f} bolum/sn"
                )

            if (
                stage_idx < len(STAGES) - 1
                and stage_episodes >= MIN_STAGE_EPISODES
                and len(stage_hist) == PROMOTE_WINDOW
                and float(np.mean(stage_hist)) >= PROMOTE_SUCCESS
            ):
                stage_idx += 1
                stage_episodes = 0
                stage_hist.clear()
                print(f"==> Asama ilerledi: {STAGES[stage_idx].name}")

            if ep % args.eval_every == 0:
                cur = evaluate(engine, eval_alg, STAGES[stage_idx], eval_seeds,
                               args.width, args.height, args.max_steps)
                gui = evaluate(engine, eval_alg, STAGES[-1], eval_seeds,
                               args.width, args.height, args.max_steps)
                print(f"   DEGERLENDIRME {STAGES[stage_idx].name}: {_fmt(cur)}")
                print(f"   DEGERLENDIRME gui: {_fmt(gui)}")
                if gui["score"] > agent.best_avg:
                    # Headless'ta best_avg, gui degerlendirme skorunu tutar.
                    agent.best_avg = gui["score"]
                    agent.save(checkpoint_best)
                    print(f"   En iyi model kaydedildi (skor {gui['score']:.3f}).")
                agent.save(checkpoint_last)
    except KeyboardInterrupt:
        print("\nDurduruldu.")
    finally:
        agent.save(checkpoint_last)
        if not os.path.exists(checkpoint_best):
            agent.save(checkpoint_best)
        print(f"Kaydedildi: {checkpoint_last}. Toplam {ep} bolum, {time.time() - t0:.0f} sn.")


if __name__ == "__main__":
    main()
