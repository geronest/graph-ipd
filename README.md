# shMARL

Simulation code for the paper:

> **The Role of Network Topology and Opponent Information in Shaping Cooperation
> in Multi-Agent Reinforcement Learning Systems**
> Seongho Son, Stephen Hailes, Mirco Musolesi

Each agent is a node in a graph and plays the two-player Iterated Prisoner's
Dilemma (IPD) against its neighbours, learning by independent deep
reinforcement learning rather than by strategy imitation. The code covers both
sets of experiments in the paper: **random matching** over Erdos-Renyi,
Watts-Strogatz and Barabasi-Albert graphs, and **partner selection**, where a
Soft Actor-Critic module chooses each agent's opponent and the dilemma-playing
state may include a binary encoding of the opponent's identity.

---

## Install

```bash
git clone https://github.com/geronest/shMARL.git
cd shMARL
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Python version.** `requirements.txt` pins the exact versions used to produce
the published results (`numpy==1.19.5`, `torch==1.10.2`, `ray[default]==2.0.1`,
`networkx==2.5.1`). Those wheels require **Python 3.7-3.9** and will not
install on newer interpreters. Use `python3.9` if you want the original
environment bit-for-bit.

The code also runs on a current stack — verified end to end on Python 3.13 with
torch 2.13, numpy 2.5, scipy 1.16, pandas 2.3, networkx 3.6 and ray 2.57. If
you install modern versions instead of the pins, everything in `bin/train_ray.py`
works; note only that RNG streams differ between numpy versions, so individual
runs will not be numerically identical to the published ones (aggregate
behaviour over repetitions should be).

## Verify your install (about 30 seconds)

```bash
pytest tests/                                                  # 9 tests
python3 bin/train_ray.py smoke    --set_id smoke --run_id check   # random matching
python3 bin/train_ray.py smoke_ps --set_id smoke --run_id check   # partner selection
```

`configs/smoke/` runs the same code paths as the paper on 4 agents for 200
rounds instead of 32 agents for 100k-200k. Results land in
`results/smoke/<config>/check/`. If those three commands pass, the environment
is good.

## Reproduce the paper

Configurations live in `configs/paper/`. `common.yaml` holds everything shared
(payoff matrix, N = 32, DQN architecture) and each experiment file overrides
what it needs. Every value is commented `[paper]` if the manuscript states it
or `[repo default]` if it does not — see the note on unstated hyperparameters
below.

```bash
# Section 4(a): random matching, 100 graphs per type, 100k rounds
python3 bin/train_ray.py random_ER --set_id paper --run_id run0
python3 bin/train_ray.py random_WS --set_id paper --run_id run0
python3 bin/train_ray.py random_BA --set_id paper --run_id run0

# Section 4(b): partner selection, 20 seeds each, 200k rounds
python3 bin/train_ray.py ps_len1_noid    --set_id paper --run_id run0
python3 bin/train_ray.py ps_len5_noid    --set_id paper --run_id run0
python3 bin/train_ray.py ps_len10_noid   --set_id paper --run_id run0
python3 bin/train_ray.py ps_len1_binary  --set_id paper --run_id run0
python3 bin/train_ray.py ps_len5_binary  --set_id paper --run_id run0
python3 bin/train_ray.py ps_len10_binary --set_id paper --run_id run0

# or run a whole set in sequence
python3 bin/run_ray_train_sets.py paper run0
```

These are long jobs. Set `num_processes` in `configs/paper/common.yaml` to the
number of repetitions you want in flight at once.

### Which config produces which figure

| Paper | Config | Notes |
|---|---|---|
| Figures 3, 4, 5 | `random_ER`, `random_WS`, `random_BA` | `p_md` vs. generation parameter and vs. average path length |
| Figure 9 | same three runs | network plots, `results/.../networks/imgs/` |
| Figure 6 | all six `ps_*` runs | final `p_mc` with 95% CI |
| Figures 7, 8 | `ps_*_noid` / `ps_*_binary` | interaction-type dynamics |
| Figures 10, 13 | `ps_*_noid` / `ps_*_binary` | `selection_*` network plots |
| Figures 11, 12 | all six `ps_*` runs | Gini coefficient dynamics |

### Hyperparameters the paper does not state

`learning_rate` (0.01) and `weight_decay` (1e-3) for both the dilemma-playing
and partner-selection networks, and `freq_refresh_target` for the SAC selector,
are not given in the manuscript. The values in `configs/paper/` are carried over
from `configs/test/default.yaml` and are marked `[repo default]`. Everything
marked `[paper]` — N = 32, payoffs, `l`, identity encoding, hidden layers
[32, 16], epsilon 0.05, gamma 0.99, Z = 16, alpha 0.01, tau 0.05, round counts,
repetition counts — is taken directly from the text.

The payoff matrix in `common.yaml` was checked against Section 2(a): it yields
T = 0.3, R = 0.2, P = -0.2, S = -0.3, satisfying T > R > P > S and 2R > T + S.
State dimensions were checked against `d_dil = 2l + ceil(log2 N)`: with N = 32
the binary identity encoding adds 5 dimensions, giving 7 / 15 / 25 for
l = 1 / 5 / 10, and 2 / 10 / 20 without identity.

> **Note on agent indexing.** `StateManager` encodes opponent identity with
> `ceil(log2(N))` bits over indices `0 .. N-1`. The paper writes the opponent
> index set as `{1, ..., N}`, which would need one more bit at N = 32; the code
> is 0-indexed.

## Repository layout

```
bin/
  train_ray.py           main entrypoint (Ray-parallel); produced all paper results
  run_ray_train_sets.py  run every config in a set, in sequence
  train.py               legacy single-process trainer - see Known issues
env/
  agentnet.py            graph construction (ER, WS, BA, complete, ring, line, tree)
  matgame.py             matrix game; payoffs indexed (a0 * 2 + a1)
  statemanager.py        state encoding: action history, opponent identity
models/
  ray/                   Ray actor versions - dqn, sac, and fixed strategies
  dqn.py, tabularq.py    single-process learners
  tft.py, rtft.py, onlyc.py, onlyd.py   fixed-strategy baselines
utils/
  results.py             summaries, confidence intervals
  visualize.py           dynamics and network plots
  analysis_net.py        graph metrics (path length, clustering, Gini)
configs/
  paper/                 the experiments in the paper
  smoke/                 fast end-to-end check
  test/                  small configs used by the test suite
```

Output is written to `results/<set_id>/<config>/<run_id>/`.

## Known issues

- **`bin/train.py` is legacy.** Its calls into `models/` and `utils/visualize`
  drifted out of sync when the Ray version became the main path, and it raises
  on `select_partner: True`. Use `bin/train_ray.py`, which is what the paper's
  results were produced with.
- Slack notifications are optional. `utils/slack_messenger.py` is imported
  lazily and only used if you set `slack_cfg` in a config. Put your token in
  `comms/<name>.yaml` — `.gitignore` excludes everything in `comms/` except the
  `default.yaml` placeholder, so real tokens cannot be committed by accident.

## Citing

See `CITATION.cff`. Please cite the paper rather than the repository.

## Licence

MIT — see `LICENSE`. Copyright is asserted for the three authors; if UCL claims
copyright in this work, adjust the holder line before making the repository
public.
