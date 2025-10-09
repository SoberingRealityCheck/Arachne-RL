# Quick Troubleshooting Guide - Training Not Improving

## 🚨 If You See No Improvement After 500k Steps

### Quick Diagnostic (5 minutes):

1. **Run diagnostics**:
   ```bash
   python diagnose_training.py
   ```
   Check: Velocity errors and orientation errors

2. **Watch visualization**:
   ```bash
   python visualize.py
   ```
   Observe: What is the robot actually doing?

3. **Check your latest training log**:
   Look at these specific values

---

## 🔍 Common Problems & Fixes

### Problem: Robot Not Moving (Frozen/Stuck)
**Symptoms**: ep_len increases but robot barely moves, small velocity errors
**Fix**:
```python
# config.py - arachne section
'FORWARD_VEL_WEIGHT': 20.0,            # Was 10.0
'HOME_POSITION_PENALTY_WEIGHT': 0.1,   # Was 0.5 (too restrictive)
```

---

### Problem: Robot Falls Immediately
**Symptoms**: ep_len stays ~100-150, high fallen_penalty in logs
**Fix**:
```python
# config.py
'UPRIGHT_REWARD_WEIGHT': 2.0,  # Was 0.5
'FALLEN_PENALTY': 100.0,       # Was 20.0
'TILT_PENALTY_WEIGHT': 0.2,    # Was 0.0 (add this if not present)
```

---

### Problem: Robot Shaking/Vibrating
**Symptoms**: High angular velocity, unstable motion
**Fix**:
```python
# config.py  
'SHAKE_PENALTY_WEIGHT': 0.2,   # Was 0.01
'ACTION_LIMIT': 0.2,           # Was 0.3 (more restrictive)
```

---

### Problem: Explained Variance Still 0
**Symptoms**: After 1M steps, explained_variance < 0.1
**Possible causes**:
1. Reward scale too large (your rewards are 20k+ per episode)
2. Episode length too variable (50 to 500 steps)
3. Value function can't learn the pattern

**Fix Option 1 - Wait longer**:
- Often improves between 1M-2M steps
- Not always a problem if other metrics improving

**Fix Option 2 - Normalize rewards** (advanced):
```python
# In train.py, wrap environment:
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(env, norm_obs=True, norm_reward=True)
```

---

### Problem: Good Early, Then Gets Worse
**Symptoms**: ep_len goes 100→300→150, rewards increase then decrease
**Causes**: 
- Learning rate too high
- Policy overfitting to early data
- Catastrophic forgetting

**Fix**:
```python
# In train.py
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048,
            learning_rate=0.0001)  # Lower from 0.0003
```

---

### Problem: Orientation Never Improves
**Symptoms**: Velocity gets better, but orientation error stays high
**Check**: Did you apply the orientation reward fix?

In `env.py`, line ~361:
```python
# Should be:
r_orientation = self.ORIENTATION_REWARD_WEIGHT * 10.0 * np.exp(-(yaw_error**2))

# NOT:
r_orientation = self.ORIENTATION_REWARD_WEIGHT * np.exp(-(yaw_error**2))
```

**If still broken after fix**: Increase weight:
```python
# config.py
'ORIENTATION_REWARD_WEIGHT': 2.0,  # Was 1.0
```

---

## 🎯 The Nuclear Option: Start Fresh

If nothing works after trying fixes:

```bash
# 1. Backup your current checkpoints (just in case)
cp -r models/arachne_checkpoints models/arachne_checkpoints_backup

# 2. Delete old checkpoints
rm -rf models/arachne_checkpoints/*.zip

# 3. Start completely fresh
python train.py
# Choose 'n' when asked about existing model
```

**Why this helps**: Old policy learned bad habits under buggy/suboptimal reward function. Fresh start with fixed rewards often works better.

---

## 📊 Expected Progress Timeline

Use this to judge if you're on track:

| Steps | ep_len_mean | ep_rew_mean | explained_variance | What should be happening |
|-------|-------------|-------------|--------------------|--------------------------|
| 0-200k | 100-150 | 15k-25k | ~0 | Learning basics, falling a lot |
| 200k-500k | 150-300 | 25k-40k | 0-0.3 | Starting to balance, survive longer |
| 500k-1M | 300-600 | 40k-70k | 0.3-0.6 | Good balance, some tracking |
| 1M-2M | 600-1000+ | 70k-120k | 0.5-0.8 | Completing episodes, good tracking |
| 2M+ | 1000-1200 | 120k+ | 0.7-0.9 | Mastery, consistent performance |

**If you're significantly below these at each checkpoint**, something is wrong.

---

## 🔧 Systematic Debugging Process

**Step 1**: After 500k steps, check ep_len_mean:
- < 150: Robot not learning to survive → Fix upright/fallen rewards
- 150-300: Robot surviving but not moving → Fix velocity rewards  
- > 300: Robot doing something right → Keep training to 1M

**Step 2**: At 1M steps, check explained_variance:
- < 0.1: Value function broken → Try reward normalization
- 0.1-0.4: Slow but learning → Keep training to 2M
- > 0.5: Good! → Just needs more time

**Step 3**: At 2M steps, check reward trends:
- Increasing: Learning is working, continue
- Flat: Might have plateaued, try curriculum learning
- Decreasing: Unstable, lower learning rate and restart

---

## 💊 Quick Fixes Reference

| Symptom | Quick Fix |
|---------|-----------|
| Won't move | `FORWARD_VEL_WEIGHT: 20.0` |
| Falls instantly | `FALLEN_PENALTY: 100.0` |
| Shakes | `SHAKE_PENALTY_WEIGHT: 0.2` |
| Wild motions | `ACTION_LIMIT: 0.2` |
| Ignores orientation | `ORIENTATION_REWARD_WEIGHT: 2.0` |
| Too cautious | `HOME_POSITION_PENALTY_WEIGHT: 0.1` |
| Unstable training | `learning_rate: 0.0001` |
| Too slow | `learning_rate: 0.0005` |

---

## 🆘 When to Ask for Help

If after trying the above you still see:
- Zero progress after 1M steps
- Explained variance stuck at 0 after 2M steps  
- Training crashes or gives NaN values
- Rewards become extremely large (> 1 million) or negative

Then there might be a deeper issue. Post your:
1. Training log output
2. Config settings
3. What you've tried
4. Output from diagnose_training.py

**Good luck! RL training requires patience but it WILL work with the right tuning! 🚀**
