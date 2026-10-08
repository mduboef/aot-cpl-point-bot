# CPL AOT on PointBot

A self-contained experiment testing whether distributional alignment via Optimal Transport can prevent Contrastive Preference Learning from collapsing onto a single dominant strategy in an environment explicitly constructed to contain multiple distinct dominant strategies.

## Motivation

Contrastive Preference Learning (CPL) maximizes the average margin between preferred and rejected trajectory segments. This expectation-based objective is largely indifferent to minority dominant strategies — behaviors that are always preferred when they appear but appear infrequently in the dataset. A CPL-trained policy can achieve low loss by collapsing onto the majority dominant strategy and assigning negligible probability to minority ones.

Alignment via Optimal Transport (AOT) replaces the average-margin objective with one that enforces first-order stochastic dominance (FSD) across score distributions. By rank-matching scores before computing the loss, AOT penalizes distributional violations rather than just average errors, which we hypothesize will cause the trained policy to maintain support for minority dominant strategies.

MetaWorld tasks used in the original CPL paper have a single natural solution path and reward-based evaluation, making them a poor test of pluralistic behavior. The PointBot environment (MODE=7) is explicitly constructed with several highly distinct dominant strategies separated by obstacle walls, making it the right proof-of-concept for this hypothesis.

## Source Codebases

- **`../aot-cpl/`** — CPL and AOT-CPL algorithm implementations (MetaWorld experiments)
- **`../PSD/`** — Pluralistic Stochastic Dominance imitation learning codebase (PointBot env)

Note that another repo is used to record pointbot demos.

## Directory Layout

```
cpl_aot_point_bot/
    env/
        pointbot.py          # PointBot gym environment (adapted from PSD paper)
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
    cpl_aot_pointbot.py      # code used in Google Colab to run experiments
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


**Hypotheses:**
1. CPL+uAOT and CPL+pAOT each achieve lower FSD loss (on their respective pairings) than baseline CPL on those same pairings.
2. AOT variants achieve higher preference accuracy on minority dominant strategy pairs,
   where CPL's collapsed policy is expected to fail.
3. Rollout diagrams show CPL collapsing to a single corridor through the obstacle maze
   while AOT variants produce trajectories spanning multiple distinct routes.

## Environment: PointBot

PointBot is a 2D continuous-control environment. The agent starts at `(-170, -130)` and must reach the origin `(0, 0)`. The state is `(x, vx, y, vy)` and the action is a 2D force `(fx, fy)`. Linear dynamics with air resistance and Gaussian noise.

## Training Protocol

The reference policy π_ref is trained once with pure BC (`python3 train.py --method ref`, configured by `configs/ref.yaml`) and saved to `models/REF_POLICY/`; rerunning it warns and overwrites that folder. Every other method loads it from there and errors out if it hasn't been trained yet.

All methods follow a shared three-phase structure:

| Phase | Steps | What trains |
|---|---|---|
| π_ref BC | `bc_steps` in `ref.yaml` | reference policy only, trained once by `--method ref` |
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


## Current State
Early training runs suggest that the distributional methods (CPL pAOT more so than the uAOT variants) are doing well in their classification-based objectives, sucessfully prefferring the "correct" element of the various pairing we construct and test them on.

evaluation metrics on the test set

Method        Orig-Pair Acc  pAOT Loss  pAOT Viol Rate  uAOT Loss  uAOT Viol Rate  Avg Reward
------------  -------------  ---------  --------------  ---------  --------------  ----------
bc            72.9%          0.6505     23.8%           0.7063     36.6%           -8686.6   
cpl           91.7%          0.0372     0.5%            0.0023     0.0%            -9934.4   
cpl_biased    75.4%          1.2867     57.7%           1.2229     66.2%           -7122.2   
cpl_paot      92.3%          0.0013     0.0%            0.0025     0.0%            -5077.1   
cpl_uaot      82.5%          5.7174     16.7%           0.0006     0.0%            -42307.6  
cpl_uaot_ref  84.4%          5.9986     14.8%           0.0001     0.0%            -8837.5   

However, looking at the rollout plots for the final policies, things look strange:
    BC                  : Sticks to the majority dominant strategy in a "messy"/inefficent way.
    CPL                 : All rollouts starts to follow minority dominant strategy then spiral far off course together in the wrong direction
    CPL Biased (λ=0.2)  : The only pluralistic looking policy, seemingly taking "messy" versions of the two dominant strategies
    CPL pAOT            : Stick to minority dominant strategy, only a little "messy" in execution with a couple overshooting the goal and having to turn around to hit it.
    CPL uAOT            : All rollouts immedately shoot in the wrong direction together.
    CPL uAOT_ref        : All rollouts start to follow majority dominant strategy but end up circling the goal at the end instead of hitting it. Lost at end.


The fact that CPL Biased demonstrates both strategies while CPL pAOT sticks to the minority dominant strategy is not what I expected. This could be the faulty of the underlying reward function I am using and the preference pairs I am starting. Tweaking that may induce the preference collapse failure mode I am excecting to see from both CPL and CPL Biased. I believe that without any sort of regularizer to downweight the "score" for the negative segment of each pairing in the loss function, the models are learning that the best way to optimize the loss is to crash the likelihood of in-distribution strategies. This is something the authors of the CPL paper point out can happen with finite datasets. I am currently working on cleaning up my code and implementing variants of distributionally aligned CPL with bias regularization between downweighting the "score" for the rejected element of each pairing. I've hear CPL works well with λ between 0.1 and 0.5.

Next Steps:
    1. DONE - Clean Up Google Drive where results are stored
    2. DONE - Add Avg Reward to Evaluation Table (colab last cell)
    3. DONE - Clean up TensorBoard logging
        Every X steps log:
            - Current eval metrics (Orig-Pair Acc, pAOT Loss, pAOT Viol Rate, uAOT Loss, uAOT Viol Rate)
            - Avg log likelihood for each action in the preffered set
                 $$\text{avg pref log prob}(\pi_\theta, \mathcal{D}^+)=\frac{1}{\sum_{\sigma^+\in\mathcal{D}^+} |\sigma^+|}\sum_{\sigma^+\in\mathcal{D}^+}\sum_{(s_t,a_t)\in \sigma^+} \log \pi_\theta(a_t | s_t)$$
            - Avg log likelihood for each action in the rejected set
            - Avg reward from 25 rollouts
    3. DONE - Rerun training
        6 methods:
            - BC
            - CPL (λ=1)
            - Biased CPL (λ=0.2)
            - CPLuAOT_ref
            - CPLuAOT
            - CPLpAOT

    4. Add bias reguarlization term (λ) to distributional CPL objective function
    5. Train distributional CPL models
        3 methods:
            - CPLuAOT_ref w λ=0.2
            - CPLuAOT w λ=0.2
            - CPLpAOT w λ=0.2
    5. Train distributional CPL pAOT with different regularization values (λ) models
        - Try λ ∈ {0.9, 0.75, 0.5, 0.1}
    6. Clean up algo code for readability
    7. Double check pAOT and uAOT eval code (λ=1 for these evals)
    8. Tweak system to induce preference collapse in CPL and demonstrate pluralistic rollouts with distributional CPL
        - Tweak the reward function used to generate preference pairs
        - Try different λ values from 0.1 to 0.5
    9. Test methods with many different starting seeds


<!-- python3 train.py --method cpl_biased
tensorboard --logdir models -->