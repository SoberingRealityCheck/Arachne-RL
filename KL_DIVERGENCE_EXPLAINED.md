# KL Divergence Spikes - What They Mean & How to Fix

## 🔍 What is Approx KL?

**KL Divergence (Kullback-Leibler divergence)** measures **how much the policy changed** between the old policy and the new policy after an update.

- **Low KL (~0.001-0.01)**: Small, safe policy updates ✅
- **Medium KL (~0.01-0.02)**: Moderate updates, still OK 🟡
- **High KL (>0.02)**: Large policy changes, potentially unstable 🔴
- **Huge spikes (>0.05)**: Policy changing drastically, training instability! 🚨

---

## ⚠️ What Do KL Spikes Mean?

### During Training:

**Large KL spikes indicate:**

1. **Policy is changing TOO FAST** 
   - Making big jumps in behavior
   - Can "forget" what it learned
   - Training becomes unstable

2. **Learning rate might be too high**
   - Taking too-large gradient steps
   - Overshooting optimal policy

3. **Reward signal suddenly changed**
   - Robot discovered new strategy
   - Fell into bad behavior pattern
   - Value function giving wrong estimates

4. **Training is becoming unstable**
   - Policy and value function disagree
   - Can lead to catastrophic forgetting
   - Performance might collapse

---

## 📊 Healthy vs Unhealthy KL Patterns

### ✅ HEALTHY (Good Training)
```
Approx KL over time:
0.005 → 0.008 → 0.006 → 0.009 → 0.007 → 0.010
```
- Stays in 0.001-0.02 range
- Small fluctuations
- Generally stable
- Policy learning smoothly

### 🟡 BORDERLINE (Watch Carefully)
```
Approx KL over time:
0.012 → 0.018 → 0.015 → 0.022 → 0.019 → 0.016
```
- Occasionally touches 0.02
- Mostly stays below 0.02
- Monitor for worsening
- Consider reducing learning rate

### 🔴 UNHEALTHY (Problem!)
```
Approx KL over time:
0.008 → 0.045 → 0.012 → 0.067 → 0.015 → 0.089
```
- **Frequent spikes above 0.02**
- **Large jumps (>0.05)**
- Training unstable
- Need immediate fix!

---

## 🛠️ How to Fix KL Spikes

### Solution 1: Reduce Learning Rate ⭐ (Most Common Fix)

**Current default:** `learning_rate=0.0003`

**Try reducing by 2-5x:**

Edit `train.py` to use:
```python
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0001,  # Reduced from 0.0003
    # ... other params
)
```

**Effect:**
- Smaller gradient steps
- More stable policy updates
- KL spikes should reduce

---

### Solution 2: Reduce Batch Size

**Current default:** `n_steps=2048`

**Try smaller batches:**
```python
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    n_steps=1024,  # Reduced from 2048
    batch_size=64,  # Add if not present
    # ... other params
)
```

**Effect:**
- More frequent, smaller updates
- Better gradient estimates
- More stable training

---

### Solution 3: Adjust Clip Range

PPO uses clipping to prevent large policy changes. If KL is spiking, the clipping might not be aggressive enough.

**Current default:** `clip_range=0.2`

**Try reducing:**
```python
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    clip_range=0.1,  # Reduced from 0.2
    # ... other params
)
```

**Effect:**
- More aggressive clipping
- Prevents large policy changes
- Should directly reduce KL

---

### Solution 4: Check Reward Scale (Critical!)

**If you JUST applied the reward fixes**, make sure you started a NEW model!

**Problem:** If you continued an old model trained under 10x rewards:
- Value function has wrong scale
- Policy gets confused
- KL spikes as it tries to reconcile conflicting signals

**Solution:** 
```bash
# Delete old checkpoints
# Start completely fresh model
python train.py
# Choose 'n' when asked to load existing model
```

---

### Solution 5: Use Adaptive KL (Advanced)

PPO has a `target_kl` parameter that can automatically adjust learning rate:

```python
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    target_kl=0.01,  # Stop updates if KL exceeds this
    # ... other params
)
```

**Effect:**
- Training stops update if KL gets too high
- Self-regulating mechanism
- Prevents catastrophic updates

---

## 🔬 Diagnosing Your Specific Case

### Check These Questions:

1. **Did you start a NEW model after the reward fixes?**
   - If NO → That's likely the problem!
   - Old model can't adapt to new reward scale
   - Solution: Start fresh

2. **When do the spikes occur?**
   - Early training (first 50k steps)? → Somewhat normal
   - After 200k+ steps? → More concerning
   - Getting worse over time? → Unstable training

3. **What's happening to episode reward when KL spikes?**
   - Reward drops? → Policy learning bad behavior
   - Reward improves? → Might be discovering new strategy (ok)
   - Reward chaotic? → Unstable training

4. **What's the explained variance?**
   - Still at 0? → Value function not learning (root cause)
   - Improving? → Policy-value mismatch during transition
   - Decreasing? → Training collapse

---

## 📈 Expected KL Behavior During Training

### Phase 1: Early Training (0-100k steps)
```
Approx KL: 0.005 - 0.015
Occasional small spikes are OK
Policy exploring different strategies
```

### Phase 2: Learning Phase (100k-500k steps)
```
Approx KL: 0.003 - 0.010
Should stabilize
Smaller KL as policy converges
Value function learning helps
```

### Phase 3: Fine-tuning (500k+ steps)
```
Approx KL: 0.002 - 0.008
Very stable, small updates
Policy mostly converged
Just optimizing details
```

---

## 🚨 When to Restart Training

**Restart if you see:**

1. **Sustained high KL (>0.03 for multiple iterations)**
2. **KL spikes getting bigger over time**
3. **Episode reward collapsing after KL spike**
4. **Explained variance going negative**
5. **Value loss exploding (>10 million after reward fix)**

**These indicate unrecoverable training instability.**

---

## ⚙️ Recommended Training Parameters for Arachne

Based on your current setup, here are stable parameters:

```python
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0001,      # Reduced for stability
    n_steps=2048,              # Good batch size
    batch_size=64,             # Process in smaller chunks
    n_epochs=10,               # Standard
    gamma=0.99,                # Standard discount
    clip_range=0.2,            # Standard (reduce to 0.1 if still spiking)
    target_kl=0.015,           # Add safety valve
    ent_coef=0.01,            # Encourage exploration
    tensorboard_log="./logs/", # Optional: for TensorBoard
)
```

---

## 📊 What Your Live Plots Should Show

### If Training is Healthy:

1. **Approx KL plot:**
   - Stays mostly green (below 0.02)
   - Small, brief spikes OK
   - Decreases over time

2. **Episode Reward plot:**
   - Generally increasing
   - Some noise is normal
   - Correlates with KL (spikes might cause dips)

3. **Explained Variance plot:**
   - Starts at 0
   - Slowly increases to 0.3-0.9
   - Should NOT decrease after improving

4. **Value Loss plot:**
   - Generally decreasing
   - Some fluctuation OK
   - Should stay in thousands (50k-500k range)

---

## 🎯 Action Plan for Your Situation

Since you're seeing **huge KL spikes**, here's what to do:

### Step 1: Check Your Current Training Stats
Look at your latest training output. Check:
- Approx KL values
- Episode reward trend
- Explained variance
- Value loss

### Step 2: Determine Root Cause

**If value loss is still in MILLIONS:**
→ You didn't start fresh after reward fix
→ Solution: Start new model

**If KL spikes but reward improving:**
→ Policy discovering new strategies
→ Solution: Reduce learning rate to 0.0001

**If KL spikes AND reward chaotic:**
→ Training unstable
→ Solution: Reduce learning rate + add target_kl

### Step 3: Apply Fix and Monitor

Watch the live plots for next 50k steps:
- KL should stabilize
- Reward should improve more smoothly
- Explained variance should start climbing

---

## 💡 Quick Reference

| KL Value | Status | Action |
|----------|--------|--------|
| 0.001-0.01 | ✅ Healthy | Continue training |
| 0.01-0.02 | 🟡 OK | Monitor closely |
| 0.02-0.03 | 🟠 Warning | Reduce learning rate |
| 0.03-0.05 | 🔴 Problem | Reduce LR + add target_kl |
| >0.05 | 🚨 Critical | Restart with lower LR |

---

## 🔧 Quick Fix Commands

### If you need to restart with better parameters:

1. Stop current training (Ctrl+C)

2. Edit `train.py` to add after the model creation:
```python
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0001,  # Lower learning rate
    target_kl=0.015,       # Safety valve
    n_steps=2048,
    batch_size=64,
)
```

3. Start fresh:
```bash
python train.py
# Choose 'y' for GUI
# Choose 'arachne'
# Choose 'n' for new model
```

4. Monitor KL plot - should stay mostly below 0.02

---

## 📚 Further Reading

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **Stable-Baselines3 Docs**: [PPO Parameters](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- **KL Divergence**: Why it matters for policy gradient methods

---

**Remember:** KL spikes are a **symptom**, not the disease. The root cause is usually:
1. Learning rate too high
2. Reward scale issues
3. Value function not learning properly

Fix the root cause, and KL will stabilize! 🎯
