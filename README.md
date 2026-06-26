# CPL AOT on PointBot

A self-contained experiment testing whether distributional alignment via Optimal Transport
can prevent Contrastive Preference Learning from collapsing onto a single dominant strategy
in an environment explicitly constructed to contain multiple distinct dominant strategies.

## Motivation

Contrastive Preference Learning (CPL) maximizes the average margin between preferred and
rejected trajectory segments. This expectation-based objective is largely indifferent to
minority dominant strategies — behaviors that are always preferred when they appear but
appear infrequently in the dataset. A CPL-trained policy can achieve low loss by collapsing
onto the majority dominant strategy and assigning negligible probability to minority ones.

Alignment via Optimal Transport (AOT) replaces the average-margin objective with one that
enforces first-order stochastic dominance (FSD) across score distributions. By rank-matching
scores before computing the loss, AOT penalizes distributional violations rather than just
average errors, which we hypothesize will cause the trained policy to maintain support for
minority dominant strategies.

MetaWorld tasks have a single natural solution path and reward-based evaluation, making them
a poor test of pluralistic behavior. The PointBot environment (MODE=7) is explicitly
constructed with several highly distinct dominant strategies separated by obstacle walls,
making it the right proof-of-concept for this hypothesis.

## Source Codebases

- **`../aot-cpl/`** — CPL and AOT-CPL algorithm implementations (MetaWorld experiments)
- **`../PSD/`** — Pluralistic Stochastic Dominance imitation learning codebase (PointBot env)

This directory contains only the files needed for the PointBot experiment. Nothing is
inherited from the full MetaWorld pipeline (no config system, no wandb, no sweep tools,
no MuJoCo-specific wrappers).

## Directory Layout

```
cpl_aot_point_bot/
    env/
        pointbot.py          # PointBot gym environment (adapted from PSD)
        pointbot_const.py    # environment constants and obstacle layout
        obstacle.py          # Obstacle and ComplexObstacle classes
    data/
        allData/             # all demos, one folder per corridor type (1-5)
            1/ ... 5/        #   features_states_actions_<i>.pkl, states_<i>.txt, visualization_<label>_<i>.png
        trainingData/        # training split (per type) + trainPreferences.pkl
        testingData/         # testing split (per type) + testPreferences.pkl
        splitData.py         # splits allData 50/50 into trainingData and testingData
        preferenceGen.py     # synthetic annotator: generates contrastive preference pairs per split
    algos/                   # BC, CPL, CPL+pAOT, CPL+uAOT loss implementations
    configs/                 # per-method yaml hyperparameter configs
    models/                  # saved runs (policy weights, eval stats, tensorboard logs)
    demoPlots/               # generated demo / rollout visualizations
    mlp.py                   # MLP policy network (Gaussian continuous actions)
    train.py                 # main training script (BC warmup + contrastive phase)
    evaluate.py              # evaluation: FSD loss, preference accuracy, reward, rollouts
    plotRollouts.py          # visualize trajectories over the PointBot obstacle map
    inspectRewards.py        # rank/plot demos by reward for a split (training | testing | all)
    requirements.txt
    README.md
```

## Data Layout and Demo Labels

`data/allData/<type>/` holds every demo for corridor type `1`–`5`. Each demo `i` is stored
as `features_states_actions_<i>.pkl` alongside a `states_<i>.txt` and a
`visualization_<label>_<i>.png` whose `<label>` is `Optimal`, `Good`, or `Bad`. Analysis and preference generation keep **Good** and **Optimal** demos and **ignore Bad** ones.

`splitData.py` copies a reproducible 50/50 split of `allData` into `data/trainingData/` and
`data/testingData/` (same per-type folder structure). `preferenceGen.py` then builds
contrastive pairs for each split, writing `trainingData/trainPreferences.pkl` and
`testingData/testPreferences.pkl`, which `train.py` consumes.

## Development Plan

### Step 1 — Understand the environment and demo format

Read the existing `.pkl` demo files to verify their structure (Optimal demos use plain
`states` / `actions` / `feature` keys; Good/Bad pairs use `Good_*` and `Bad_*` keys) and
inspect the MODE=7 obstacle layout. Confirm that the demos exhibit genuine multi-strategy
structure by checking that trajectories follow visually distinct paths through the obstacle
maze.

**Deliverable:** notes on demo schema, obstacle geometry, and confirmation of multi-strategy
structure.

### Step 2 — Plot demonstrations

Implement `plot_rollouts.py` to visualize training and test demos over the PointBot
obstacle map. Each rollout should also print basic evaluation statistics:

- total trajectory length (number of steps)
- timesteps spent inside an obstacle region
- cumulative ground-truth reward

This establishes the visual and quantitative baseline before any learning.

**Deliverable:** `plot_rollouts.py`, working plots of all training and test demos with
printed per-trajectory stats.

### Step 3 — Build `preferenceGen.py`

Implement a synthetic annotator that converts the demonstration set into a dataset of
contrastive preference pairs `(σ⁺, σ⁻)`. Preference labels are assigned by comparing
cumulative ground-truth reward. The annotator should be configurable so that minority
dominant strategies appear in the data but at a lower rate than the majority strategy,
matching the failure mode described in the paper. I am unsure if one annotator can
induce the sort of diverse dominant strategies we are looking for or if we will need
multiple with different reward functions.

The output is a preference dataset `D_pref = {(σ₁⁺, σ₁⁻), ..., (σₙ⁺, σₙ⁻)}` stored
as a flat list of paired trajectory dicts.

Since all the demos in the current training and testing data are all considered "good"
we may have to take some of the Point Bot synthetic demo generation functionality from
the PSD codebase and generate some noisy otherwise suboptimal demos to add to our
database of demos. We may need to only generate preference between demos that have the
same strategy/reward function so that we get multiple distinct dominant strategies in our
data.

**Deliverable:** `data/preferenceGen.py`, a saved preference dataset.

### Step 4 — Preference accuracy evaluation

Implement the secondary evaluation metric: the fraction of annotator preference pairs for
which the policy assigns a higher segment score to the preferred trajectory than to the
rejected one.

```
score_π(σ) = Σ_t γ^t α log π(a_t | s_t)
preference_accuracy = mean[ score_π(σ⁺) > score_π(σ⁻) ]
```

This function lives in `evaluate.py` and is shared across all methods. It is evaluated
on the original annotator pairs regardless of which training objective was used.

**Deliverable:** `evaluate.py` with `preference_accuracy(policy, preference_dataset)`.

### Step 5 — Bare-bones BC training

Implement `train.py` with a behavior cloning phase that trains on all trajectories in the
preference dataset (both preferred and rejected segments). The policy network should be an
MLP that takes a state observation and outputs a Gaussian action distribution.

After training, the script should roll out the learned policy, print evaluation stats,
and call `plot_rollouts.py` to visualize the trajectories.

The script structure should make it easy to add a contrastive phase later: the BC phase
trains for a configurable number of steps, then hands off to whatever contrastive loss
is enabled.

**Deliverable:** `train.py` with `--method bc`, working rollouts and plots.

### Step 6 — CPL

After the BC warmup, switch to the CPL contrastive loss on the annotator preference pairs:

```
L(θ) = (1/n) Σᵢ -log( exp(score_θ(σᵢ⁺)) / (exp(score_θ(σᵢ⁺)) + exp(score_θ(σᵢ⁻))) )
```

Enable via `--method cpl`.

**Deliverable:** `--method cpl` working end-to-end with evaluation and plots.

### Step 7 — CPL (biased)

Same as CPL but with asymmetric bias regularization (λ = 0.5):

```
logit = score_θ(σ⁺) - λ · score_θ(σ⁻)
```

Enable via `--method cpl_biased`.

**Deliverable:** `--method cpl_biased` working end-to-end.

### Step 8 — CPL+pAOT (with FSD loss as shared eval metric)

Implement the paired AOT variant. The loss operates on per-pair margins under π_θ and a
frozen reference policy π_ref. Before contrastive training begins, π_ref is trained with
BC for a configurable number of steps.

```
u_θⁱ = score_θ(σᵢ⁺) - score_θ(σᵢ⁻)     # policy margin
v_refⁱ = score_ref(σᵢ⁺) - score_ref(σᵢ⁻) # reference margin

sort both independently (1D OT via northwest corner)
L(θ) = (1/n) Σᵢ -log( exp(u_θ^(i)) / (exp(u_θ^(i)) + exp(v_ref^(i))) )
```

**Important:** The sorting and FSD loss computation should be written as standalone
functions in `evaluate.py`, not buried inside the training loop. This is the primary
evaluation metric — CPL and BC policies will also be scored with it.

FSD loss for pAOT pairings:
```
FSD_pAOT(π) = mean( ReLU(v_ref^(i) - u_θ^(i)) )
```
where margins are OT-sorted independently before comparison.

Enable via `--method cpl_paot`.

**Deliverable:** `--method cpl_paot` working end-to-end; `evaluate.py` exports
`fsd_loss_paot(policy, ref_policy, preference_dataset)` usable on any policy.

### Step 9 — CPL+uAOT (with FSD loss as shared eval metric)

Implement the unpaired AOT variant. The loss operates on raw segment scores pooled across
all preferred segments and all rejected segments independently.

```
U = { score_θ(σᵢ⁺) } for all i    # preferred scores
V = { score_θ(σᵢ⁻) } for all i    # rejected scores

sort U and V independently (1D OT)
L(θ) = (1/n) Σᵢ -log( exp(U^(i)) / (exp(U^(i)) + exp(V^(i))) )
```

The reference-policy variant replaces raw scores with log-ratio scores:
```
score_θ(σ) → Σ_t γ^t α (log π_θ(a_t|s_t) - log π_ref(a_t|s_t))
```

FSD loss for uAOT pairings:
```
FSD_uAOT(π) = mean( ReLU(V^(i) - U^(i)) )
```
where U and V are OT-sorted independently before comparison.

Enable via `--method cpl_uaot` and `--method cpl_uaot_ref`.

**Deliverable:** `--method cpl_uaot` and `--method cpl_uaot_ref` working end-to-end;
`evaluate.py` exports `fsd_loss_uaot(policy, preference_dataset)` usable on any policy.

### Step 10 — Comparative analysis

Run all five methods (BC, CPL, CPL-biased, CPL+pAOT, CPL+uAOT) and compare on:

| Metric | Description |
|---|---|
| FSD loss (pAOT pairings) | primary — stochastic dominance violations on per-pair margins |
| FSD loss (uAOT pairings) | primary — stochastic dominance violations on pooled score distributions |
| Preference accuracy | secondary — fraction of annotator pairs correctly ordered |
| Ground-truth reward | tertiary — average cumulative reward over rollouts |
| Rollout diagrams | visual — do policies cover all dominant strategies or collapse? |

**Hypotheses:**
1. CPL+uAOT and CPL+pAOT each achieve lower FSD loss (on their respective pairings) than
   baseline CPL on those same pairings.
2. AOT variants achieve higher preference accuracy on minority dominant strategy pairs,
   where CPL's collapsed policy is expected to fail.
3. Rollout diagrams show CPL collapsing to a single corridor through the obstacle maze
   while AOT variants produce trajectories spanning multiple distinct routes.

## Environment: PointBot MODE=7

PointBot is a 2D continuous-control environment. The agent starts at `(-170, -130)` and
must reach the origin `(0, 0)`. The state is `(x, vx, y, vy)` and the action is a 2D
force `(fx, fy)`. Linear dynamics with air resistance and Gaussian noise.

MODE=7 uses a complex multi-wall obstacle layout that creates several physically separated
corridors between the start and goal. Each corridor corresponds to a distinct dominant
strategy. The environment is designed so that navigating any corridor successfully achieves
the goal, but the corridors require qualitatively different trajectories.

The ground-truth reward for a trajectory is the negative of the sum of step costs:
```
step_cost(s, a) = ||s - GOAL_STATE|| + collision_cost(s)
```

## Training Protocol

All methods follow a shared three-phase structure:

| Phase | Steps | What trains |
|---|---|---|
| π_ref BC | `ref_bc_steps` | reference policy only (pAOT/uAOT-ref only) |
| π_θ BC warmup | `theta_bc_steps` | policy under BC |
| Contrastive | `contrastive_steps` | policy under method-specific loss |

**Hyperparameters (following MetaWorld setup):**
- Architecture: MLP `[256, 256]` hidden dims
- Optimizer: Adam, lr = 1e-4
- Entropy temperature: α = 0.1
- Discount: γ = 1.0 (matching CPL paper convention)
- Bias (CPL-biased): λ = 0.5

## Requirements

```
torch
gym
numpy
matplotlib
pickle
```

See `requirements.txt` for pinned versions.

<!-- python3 train.py --method cpl_biased
tensorboard --logdir models -->