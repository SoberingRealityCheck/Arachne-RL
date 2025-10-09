# Understanding PPO Training Logs - Explained Simply

## Your Training Output Decoded

```
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 119          |
|    ep_rew_mean          | 2.46e+04     |
| time/                   |              |
|    fps                  | 284          |
|    iterations           | 109          |
|    time_elapsed         | 785          |
|    total_timesteps      | 223232       |
| train/                  |              |
|    approx_kl            | 0.0048693214 |
|    clip_fraction        | 0.0111       |
|    clip_range           | 0.2          |
|    entropy_loss         | -25          |
|    explained_variance   | -1.19e-07    |
|    learning_rate        | 0.0003       |
|    loss                 | 5.48e+06     |
|    n_updates            | 1080         |
|    policy_gradient_loss | -0.0191      |
|    std                  | 0.972        |
|    value_loss           | 1.13e+07     |
------------------------------------------
```

---

## 📊 Section 1: ROLLOUT (How the robot is performing)

### `ep_len_mean: 119`
**What it means**: Average episode length (how many timesteps before episode ends)
- Your episodes last 119 steps on average
- At 240Hz physics, that's about **0.5 seconds** (119/240)
- With `action_skip=10`, that's about **12 actions**

**Good or bad?**
- Low values (< 100): Robot is falling quickly or failing fast
- Medium values (100-500): Robot is surviving but not mastering
- High values (1000+): Robot is staying alive and completing task

**Your case**: 119 is pretty low - robot is falling or failing within half a second. This is normal early in training!

### `ep_rew_mean: 2.46e+04` (24,600)
**What it means**: Average total reward per episode
- Higher = better performance
- This is the sum of all step rewards before episode ends

**What to watch for**:
- Should **increase over time** if learning is working
- Sudden drops = learning instability
- Plateau = need better reward shaping or hyperparameters

**Your case**: 24,600 reward in 119 steps = ~207 reward per step. This seems reasonable if your rewards are in that range.

---

## ⏱️ Section 2: TIME (Training speed metrics)

### `fps: 284`
**What it means**: Frames (simulation steps) per second
- How fast your training is running
- **This is your training speed bottleneck**

**Interpretation**:
- < 100 fps: Slow (but ok for complex robots)
- 100-300 fps: Normal for CPU training
- 300-1000 fps: Good (CPU or small GPU)
- 1000+ fps: Fast (good GPU)

**Your case**: 284 fps is decent for CPU training. With GPU, you could get 600-800 fps.

### `iterations: 109`
**What it means**: Number of training iterations completed
- One iteration = collect data, then update policy
- You collect `n_steps=2048` steps per iteration
- Then train on that batch

**Your case**: 109 iterations × 2048 steps = 223,232 total steps ✓

### `total_timesteps: 223,232`
**What it means**: Total simulation steps since training started
- This is your **progress metric**
- You set `total_timesteps=5,000,000`, so you're at **4.5%** done

### `time_elapsed: 785` (seconds)
**What it means**: Training has been running for 785 seconds = **13 minutes**

**Calculate ETA**:
- 785 seconds for 223k steps
- Need 5M steps total
- ETA: (5M / 223k) × 785 = **~58 minutes** to completion

---

## 🧠 Section 3: TRAIN (The actual learning metrics)

### `approx_kl: 0.0048693214`
**What it means**: KL divergence - how much the policy changed this update
- Measures difference between old policy and new policy
- PPO tries to keep this small for stable learning

**Good values**:
- 0.001 - 0.02: Good, stable learning
- > 0.05: Policy changing too fast (might be unstable)
- < 0.001: Policy barely changing (might be stuck)

**Your case**: 0.0049 is perfect! Nice stable updates.

---

### `clip_fraction: 0.0111` (1.11%)
**What it means**: Percentage of policy updates that got "clipped"
- PPO clips large updates to prevent drastic changes
- Higher = more aggressive updates being limited

**Good values**:
- 0.0 - 0.1 (0-10%): Normal, healthy learning
- 0.1 - 0.3 (10-30%): Moderate clipping, still ok
- > 0.3 (30%+): Too much clipping, policy wants to change a lot

**Your case**: 1.11% is excellent - very stable, gentle updates.

---

### `clip_range: 0.2`
**What it means**: The clipping threshold for PPO
- This is a **hyperparameter** you set (usually 0.2)
- Controls how much policy can change per update
- Not something you need to monitor, just reference

---

### `entropy_loss: -25`
**What it means**: Measure of policy randomness (exploration)
- **More negative = less random (more deterministic)**
- **Less negative = more random (more exploration)**

**What's happening**:
- Early training: Higher (less negative) - more exploration
- Late training: Lower (more negative) - more exploitation

**Good values**:
- Start: -5 to -15 (lots of exploration)
- Middle: -15 to -25 (balanced)
- End: -25 to -35 (mostly exploitation)

**Your case**: -25 suggests the policy is becoming fairly confident. For continuous control (your robot), this is normal.

---

### ⚠️ `explained_variance: -1.19e-07` (NEAR ZERO!)

**This is your key question!**

**What it means**: How well the value function predicts actual returns
- Ranges from -∞ to 1.0
- **1.0 = Perfect prediction**: "I knew exactly what reward I'd get!"
- **0.0 = No prediction**: "I'm just guessing randomly"
- **Negative = Worse than guessing**: "My predictions are actively wrong"

**The value function** is the critic that estimates "how good is this state?"
- It learns to predict future rewards
- Helps the policy learn faster by providing better learning signals

**Why is yours near zero?**

### 🔍 Reason 1: Early Training (MOST LIKELY)
At the start of training:
- Value function hasn't learned anything yet
- It's making random predictions
- As training progresses, this should improve

**What to expect**:
- First 50k-100k steps: Explained variance near 0 or negative
- 100k-500k steps: Should start increasing (0.1 to 0.5)
- 500k+ steps: Should be 0.5 to 0.9

**You're only at 223k steps, so 0 is NORMAL!**

### 🔍 Reason 2: Reward Scale Issues
Your rewards are HUGE: 24,600 per episode!
- Large reward values can make it hard for value function to learn
- Value function needs to predict these big numbers
- Normalization might help

**Check your reward components**:
```python
# From your code:
r_lin_vel = 10.0 * exp(...)           # Max ~10
r_orientation = 1.0 * 10.0 * exp(...) # Max ~10
r_survival = 0.02 * 119 = 2.38        # Small
r_fallen = -20.0                       # When fallen
```

Wait... 24,600 reward but individual rewards are ~10-20?
That means **you're accumulating over 119 steps with action_skip=10**!
119 steps × ~200 reward/step = 23,800 ✓

**This is fine, but the value function needs time to learn these patterns.**

### 🔍 Reason 3: High Variance Episodes
If episode lengths and rewards vary a lot:
- Some episodes: 50 steps, 10k reward
- Other episodes: 200 steps, 40k reward
- Value function can't predict well

**Solution**: Give it time. Should improve by 500k steps.

---

### `learning_rate: 0.0003`
**What it means**: How big the learning steps are
- Higher = faster learning (but less stable)
- Lower = slower learning (but more stable)

**Your case**: 0.0003 is the default and works well.

---

### `loss: 5.48e+06` (5,480,000)
**What it means**: Combined policy + value loss
- This should **generally decrease** over time
- But can fluctuate, especially with RL

**Your case**: 5.48 million seems high, but this depends on your reward scale. With rewards of 24k per episode, large losses are expected. Watch for the **trend**, not absolute value.

---

### `policy_gradient_loss: -0.0191`
**What it means**: The policy improvement signal
- Negative is normal (it's actually a "gain" not a "loss")
- Larger magnitude = bigger policy updates

**Your case**: -0.019 is small and stable. Good!

---

### `std: 0.972`
**What it means**: Standard deviation of the policy's action distribution
- How "spread out" the action choices are
- **Higher = more exploration, more random**
- **Lower = more exploitation, more deterministic**

**What's happening**:
- Start: std = 1.0 (default, random actions)
- Training: Gradually decreases
- End: std = 0.5-0.8 (confident but some exploration)

**Your case**: 0.972 is barely changed from 1.0. You're still exploring randomly. This is normal at 223k steps!

---

### `value_loss: 1.13e+07` (11,300,000)
**What it means**: How wrong the value function's predictions are
- Should decrease as value function learns
- Related to explained variance

**Your case**: High value loss + low explained variance = value function hasn't learned yet. Normal at this stage!

---

## 🎯 What Should You Watch?

### ✅ Healthy Training Signs:
1. ✅ **`ep_len_mean` increasing** (119 → 500 → 1000+)
2. ✅ **`ep_rew_mean` increasing** (24k → 30k → 40k+)
3. ✅ **`approx_kl` stays 0.001-0.02** ✓ (yours is 0.0049)
4. ✅ **`clip_fraction` stays 0-0.2** ✓ (yours is 0.011)
5. ✅ **`explained_variance` increases over time** (0 → 0.5 → 0.8)
6. ✅ **`fps` stays consistent** ✓ (yours is 284)

### ⚠️ Warning Signs:
- ❌ `approx_kl` > 0.05 (policy changing too fast)
- ❌ `ep_len_mean` decreasing (getting worse!)
- ❌ `explained_variance` stays negative after 1M steps
- ❌ `fps` dropping (memory leak or system issues)

---

## 📈 Your Current Status

**After 223k steps (13 minutes, 4.5% done):**

| Metric | Value | Status |
|--------|-------|--------|
| Episode length | 119 steps | 🟡 Low (robot falling quickly) |
| Episode reward | 24,600 | 🟢 Reasonable |
| FPS | 284 | 🟢 Good for CPU |
| KL divergence | 0.0049 | 🟢 Perfect |
| Clip fraction | 1.11% | 🟢 Excellent |
| Explained variance | ~0 | 🟡 Expected at this stage |
| Std deviation | 0.972 | 🟢 Still exploring |

**Verdict**: Everything looks **normal and healthy** for early training!

---

## 🔮 What to Expect Next

### By 500k steps:
- Episode length: 200-400
- Episode reward: 35k-50k
- Explained variance: 0.3-0.6
- Std: 0.85-0.95

### By 1M steps:
- Episode length: 500-800
- Episode reward: 60k+
- Explained variance: 0.5-0.8
- Std: 0.75-0.90

### By 3M steps:
- Episode length: 1000+ (max episode)
- Episode reward: 100k+
- Explained variance: 0.7-0.9
- Std: 0.65-0.85

---

## 🎓 Key Takeaways

1. **Explained variance of 0 at 223k steps is COMPLETELY NORMAL**
   - Value function needs time to learn
   - Should improve by 500k-1M steps
   - Don't worry unless it's still 0 after 2M steps

2. **Your training looks healthy!**
   - Stable KL and clip fraction
   - Consistent FPS
   - No warning signs

3. **Be patient**
   - You're only 4.5% done (223k / 5M)
   - Real improvements come after 500k-1M steps
   - With fixed orientation reward, should see better results

4. **Focus on these metrics**:
   - `ep_len_mean` increasing = robot surviving longer
   - `ep_rew_mean` increasing = robot performing better
   - `explained_variance` improving = value function learning

---

## 📊 Quick Reference

**Good news**: All your metrics look normal for early training!
**When to worry**: If explained variance is still ~0 after 2M steps
**ETA to completion**: ~55 more minutes (with current FPS)
**Expected improvement**: Should see much better performance by 1M steps

Keep training! 🚀

---

## 🚨 Troubleshooting: No Improvement After 500k Steps

If you reach 500k steps and see:
- ❌ Episode length still ~100-150 (not increasing)
- ❌ Explained variance still ~0
- ❌ Rewards not increasing or very noisy
- ❌ Robot still falling immediately

### Here's what to check:

---

### 1️⃣ **Check Your Reward Values Are Reasonable**

Run the diagnostic to see actual reward breakdown:

```bash
python diagnose_training.py
```

**Look for:**
- Are individual step rewards in reasonable ranges? (-10 to +100)
- Is one reward component dominating everything?
- Are penalties too harsh? (causing instant episode termination)

**Example problem**: If `r_fallen = -20` but total reward per step is only ~5, then falling gives you 4 good steps worth of penalty. Not harsh enough! Should be more like -100 to -200.

---

### 2️⃣ **Verify Reward Components Are Balanced**

From your training logs, check the reward breakdown prints (every 240 steps):

```
Step Reward Breakdown: Vel Track: 7.50, Fwd Bonus: 0.50, 
Upright: 0.40, Survival: 0.20, Orient: 8.20, ...
```

**Good signs:**
- ✅ Multiple rewards contributing (velocity ~8, orientation ~8, upright ~0.5)
- ✅ All values in similar magnitude (not one 1000x bigger)
- ✅ Penalties are noticeable but not overwhelming

**Bad signs:**
- ❌ One reward dominates: "Vel Track: 0.1, Orient: 99.9" (orientation drowning out velocity)
- ❌ Penalties too harsh: "Fallen: -1000, everything else: ~10"
- ❌ Rewards too small: "All values < 0.01"

**Fix**: Adjust weights in `config.py`:
```python
'FORWARD_VEL_WEIGHT': 10.0,      # Lower if too dominant
'ORIENTATION_REWARD_WEIGHT': 1.0, # Raise if too weak (gets 10x multiplier)
'FALLEN_PENALTY': 100.0,         # Raise if robot doesn't care about falling
```

---

### 3️⃣ **Check if Robot is Actually Moving**

Watch a visualization:

```bash
python visualize.py
```

**What to look for:**
- Is robot just frozen? → Rewards might not incentivize movement
- Is robot vibrating/shaking? → Shake penalty too weak or action_limit too large
- Is robot trying but falling? → Upright reward too weak
- Is robot moving but wrong direction? → Velocity reward not working

**Specific fixes:**

**If robot is frozen (not moving):**
```python
# In config.py, increase movement incentive:
'FORWARD_VEL_WEIGHT': 20.0,  # Was 10.0
'HOME_POSITION_PENALTY_WEIGHT': 0.1,  # Was 0.5 (reduce - it's keeping robot too still)
```

**If robot is shaking:**
```python
'SHAKE_PENALTY_WEIGHT': 0.2,  # Was 0.01 (increase)
'ACTION_LIMIT': 0.2,  # Was 0.3 (more restrictive)
```

**If robot falls immediately:**
```python
'UPRIGHT_REWARD_WEIGHT': 2.0,  # Was 0.5 (increase)
'FALLEN_PENALTY': 50.0,  # Was 20.0 (increase)
```

---

### 4️⃣ **Check Observation Space is Informative**

Your policy needs to know:
- ✅ Current velocity (has it)
- ✅ Target velocity (has it)
- ✅ Current orientation (has it via quaternion)
- ✅ Target orientation (has it)
- ✅ Joint positions and velocities (has it)

This should be fine based on your `_get_obs()` function.

---

### 5️⃣ **Learning Rate Issues**

If training is too slow or unstable:

**Too slow (no progress after 1M steps):**
```python
# When creating model in train.py:
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, 
            learning_rate=0.0005)  # Was 0.0003, try higher
```

**Too unstable (rewards bouncing wildly):**
```python
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, 
            learning_rate=0.0001)  # Was 0.0003, try lower
```

---

### 6️⃣ **Try Curriculum Learning**

If the task is too hard, start easier:

**Phase 1: Just stay upright (0-500k steps)**
```python
# In config.py:
'FORWARD_VEL_WEIGHT': 1.0,        # Very low
'ORIENTATION_REWARD_WEIGHT': 0.1, # Very low  
'UPRIGHT_REWARD_WEIGHT': 5.0,     # Very high
'FALLEN_PENALTY': 100.0,          # High
```

Once robot can stay upright reliably (ep_len > 500), **then** enable full task:

**Phase 2: Add velocity tracking (500k-2M steps)**
```python
'FORWARD_VEL_WEIGHT': 10.0,       # Normal
'ORIENTATION_REWARD_WEIGHT': 1.0, # Normal
'UPRIGHT_REWARD_WEIGHT': 0.5,     # Back to normal
```

---

### 7️⃣ **Check Action Space Limitations**

Your `ACTION_LIMIT = 0.3` means joints can only move ±18° from home position.

**Too restrictive?** Robot can't reach good poses:
```python
'ACTION_LIMIT': 0.5,  # Allow more freedom
```

**Too loose?** Robot making wild movements:
```python
'ACTION_LIMIT': 0.2,  # More restrictive
```

Test by watching visualize.py and seeing if joints look constrained.

---

### 8️⃣ **Network Architecture Too Small?**

Default MlpPolicy is [64, 64] (two layers, 64 neurons each). For complex robots:

```python
from stable_baselines3 import PPO

policy_kwargs = dict(net_arch=[256, 256])  # Bigger network

model = PPO("MlpPolicy", env, verbose=1, n_steps=2048,
            policy_kwargs=policy_kwargs)
```

**When to try this**: If robot seems stuck in local optimum after 2M steps.

---

### 9️⃣ **Physics Timestep Issues**

Check your simulation settings in `env.py`:

```python
self.time_step = 1.0 / 240.0  # 240 Hz physics
self.action_skip = 10          # Apply action for 10 steps
```

**Problem**: If action_skip is too high, robot doesn't react fast enough.

**Try**:
```python
self.action_skip = 5  # More responsive control
```

---

### 🔟 **Start Fresh with Fixed Rewards**

If you've been training on the old (buggy) reward function:

```bash
# Delete old checkpoints
rm -rf models/arachne_checkpoints/*

# Start fresh with new rewards
python train.py
# Choose 'n' to start from scratch
```

The old policy learned bad habits under wrong incentives. Starting fresh often works better than continuing!

---

## 🎯 Decision Tree: What To Do

```
After 500k steps, if not improving:

1. Check rewards (diagnose_training.py)
   - Balanced? → Continue to step 2
   - Imbalanced? → Adjust weights, restart training

2. Watch visualization (visualize.py)  
   - Robot trying but falling? → Increase upright/fallen weights
   - Robot not moving? → Reduce home penalty, increase velocity weight
   - Robot shaking? → Increase shake penalty, reduce action limit
   - Robot looks good but metrics bad? → Give it more time (go to 1M steps)

3. Check explained variance at 1M steps
   - Still ~0? → Rewards might be too large/small, try normalizing
   - Improving (0.2-0.5)? → Keep training to 2M
   - Good (0.6+)? → Just needs more time

4. After 2M steps, still no improvement?
   - Try curriculum learning (step 6 above)
   - Try bigger network (step 8 above)  
   - Check action_limit (step 7 above)
   - Consider starting completely fresh
```

---

## 📞 Quick Diagnostic Checklist

At 500k steps, run through this:

- [ ] Run `diagnose_training.py` and check velocity/orientation errors
- [ ] Run `visualize.py` and watch robot behavior for 30 seconds
- [ ] Check training logs: is `approx_kl` still 0.001-0.02? (should be)
- [ ] Check training logs: is `clip_fraction` still < 0.2? (should be)
- [ ] Check: has `ep_len_mean` increased at all? (even 119→150 is progress)
- [ ] Check: is `ep_rew_mean` trending upward? (even slowly)
- [ ] Check: is `explained_variance` above 0.1? (any improvement is good)

**If most are ❌**: Follow fixes above and restart training
**If most are ✅**: Be patient, keep training to 1M-2M steps

---

## 💡 Pro Tip: Keep Notes

Track your experiments:

```
Experiment 1 (500k steps):
- Config: FORWARD_VEL_WEIGHT=10, ORIENT=1, ACTION_LIMIT=0.3
- Result: ep_len=130, no orientation improvement
- Decision: Increase ORIENT to 2.0, restart

Experiment 2 (500k steps):  
- Config: FORWARD_VEL_WEIGHT=10, ORIENT=2, ACTION_LIMIT=0.3
- Result: ep_len=280, both improving!
- Decision: Continue to 2M steps
```

This helps you learn what works for YOUR specific robot!

---

**Remember**: RL training is inherently noisy and requires patience. But if you see ZERO improvement after 500k-1M steps, something is probably misconfigured. Use the diagnostics above to debug! 🔧
