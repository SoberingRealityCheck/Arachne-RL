# Real-Time Training Plots Guide

## 📊 Live Training Visualization

Your training script now includes **live plotting** that shows training progress in real-time!

---

## 🎯 What Gets Plotted

The plot window shows **8 key metrics** in a 2x4 grid:

### Row 1: Performance Metrics
1. **Episode Length** - How long episodes last (should increase)
2. **Episode Reward** - Total reward per episode (should increase)
3. **Explained Variance** - How well value function predicts (0→1, should increase)
4. **Value Loss** - Value function error (should decrease)

### Row 2: Training Health Metrics
5. **Policy Loss** - Policy gradient loss (should stay stable)
6. **Approx KL** - Policy change size (should stay 0.001-0.02)
7. **Training Speed (FPS)** - Steps per second (monitors performance)
8. **Progress** - Total timesteps completed

---

## 🚀 How to Use

### With GUI (Interactive Mode):

```bash
python train.py
# Choose 'y' for GUI
```

**What happens:**
- ✅ Training starts
- ✅ Plot window opens automatically after first 2048 steps
- ✅ Plots update every 2048 steps (~7 seconds at 290 fps)
- ✅ You can watch training progress in real-time!
- ✅ When training finishes, plot window stays open

**Tips:**
- Resize the plot window to see details better
- Plot updates don't slow down training significantly
- You can still see terminal output alongside plots

### Headless Mode (No GUI):

```bash
python train.py
# Choose 'n' for GUI
```

**What happens:**
- ✅ Training runs without GUI (faster)
- ✅ Plots are **saved as images** every 50k steps
- ✅ Images saved to `./training_plots/`
- ✅ Files named: `training_metrics_50000.png`, `training_metrics_100000.png`, etc.
- ✅ Final plot saved when training completes

**Good for:**
- Cloud training
- Overnight training
- Maximum performance (no GUI overhead)

---

## 📈 What to Look For

### Healthy Training Signs:

#### Episode Length (Top Left)
- ✅ **Steady upward trend**: 100 → 200 → 500 → 1000+
- ✅ Some fluctuation is normal
- ❌ Flat or decreasing = robot not improving

#### Episode Reward (Top 2nd)
- ✅ **Upward trend**: 20k → 30k → 50k → 100k+
- ✅ Can be noisy, look at overall direction
- ❌ Flat after 500k steps = need to adjust config

#### Explained Variance (Top 3rd)
- ✅ **Starts near 0, increases**: 0 → 0.3 → 0.6 → 0.9
- 🟡 Red dashed line at 0 (bad)
- 🟢 Green dashed line at 1 (perfect)
- ⏳ Takes 500k-1M steps to improve
- ❌ Still near 0 after 1M steps = value function not learning

#### Value Loss (Top Right)
- ✅ **General downward trend** (can fluctuate)
- ✅ Smaller is better
- ❌ Increasing over time = training instability

#### Policy Loss (Bottom Left)
- ✅ **Stays relatively stable** (negative is normal)
- ✅ Small fluctuations ok
- ❌ Large spikes or wildly changing = unstable learning

#### Approx KL (Bottom 2nd)
- ✅ **Should stay in green zone**: 0.001 - 0.02
- 🟠 Orange line at 0.02 (warning threshold)
- 🔴 Red line at 0.05 (danger - policy changing too fast)
- ❌ Consistently above 0.02 = learning rate too high

#### FPS (Bottom 3rd)
- ✅ **Stays consistent**: Should hover around same value (e.g., 280-300)
- ❌ Declining = memory leak or system issues
- ℹ️ Headless is slightly faster than GUI mode

#### Progress (Bottom Right)
- ℹ️ Just shows total timesteps (linear increase)
- Useful for seeing overall progress at a glance

---

## 🎨 Reading the Plots

### Plot Colors & Lines:
- **Blue solid line**: Your data
- **Red dashed line**: Warning/threshold (bad)
- **Green dashed line**: Target/threshold (good)
- **Orange dashed line**: Warning threshold

### Example: Good Training
```
After 500k steps:
✅ Episode Length: 100 → 400 (increasing!)
✅ Episode Reward: 24k → 45k (increasing!)
✅ Explained Variance: 0 → 0.4 (improving!)
✅ Value Loss: 1.1e7 → 5e6 (decreasing!)
✅ Approx KL: Stays around 0.005 (stable!)
```

### Example: Problem Training
```
After 500k steps:
❌ Episode Length: Still ~100-120 (flat!)
❌ Episode Reward: Bouncing 20k-30k (noisy, no trend)
❌ Explained Variance: Still ~0 (not learning)
❌ Approx KL: Spiking above 0.02 (unstable)
→ Need to adjust config and restart
```

---

## 🔧 Customizing Plot Behavior

### Change Update Frequency

In `train.py`, find this line:
```python
plot_callback = LivePlottingCallback(
    plot_freq=2048,  # Update every 2048 steps
    max_points=500,  # Keep last 500 data points
)
```

**Update more often** (but slower):
```python
plot_freq=1024,  # Every 1024 steps
```

**Update less often** (but faster):
```python
plot_freq=4096,  # Every 4096 steps
```

### Keep More History

```python
max_points=1000,  # Show last 1000 data points (more memory)
```

### Change Save Frequency (Headless)

```python
plot_callback = LivePlottingCallbackNoGUI(
    save_freq=100000,  # Save every 100k steps instead of 50k
)
```

---

## 💾 Saving Plots

### GUI Mode:
The plot window stays open when training completes. You can:
- **Manually save**: Click the save icon in plot window
- **Screenshot**: Alt+PrtScn to capture
- **Auto-save**: Close window when done (asks to save)

### Headless Mode:
Plots automatically saved to `./training_plots/` as PNG files.

**View them:**
```bash
# On Windows
start training_plots\training_metrics_500000.png

# Or open folder
explorer training_plots
```

---

## 🐛 Troubleshooting

### "Plot window doesn't open"
- **Solution**: Wait for first 2048 steps (~7 seconds)
- Check terminal for errors
- Try headless mode if GUI issues persist

### "Plot window freezes"
- **Solution**: This is normal - it freezes during training steps
- Unfreezes when updating (every 2048 steps)
- Don't close it forcefully!

### "Training slower with plots"
- **Impact**: Minimal (~5-10% overhead)
- **Solution**: Use headless mode for maximum speed
- **Or**: Increase `plot_freq` to update less often

### "Memory usage increasing"
- **Solution**: Reduce `max_points` from 500 to 250
- Old data points are automatically removed

### "matplotlib not installed"
```bash
conda activate arachne
pip install matplotlib
```

---

## 📊 Example Use Cases

### Quick Check During Training
```
1. Start training with GUI
2. Watch plots for 5 minutes
3. See if metrics trending correctly
4. If not, Ctrl+C and adjust config
5. Restart with new config
```

### Long Training Session
```
1. Start training in headless mode
2. Let it run overnight
3. Next day, check saved plots
4. See progress over time
5. Decide if continuing or adjusting
```

### Comparing Experiments
```
1. Train with config A → saves plots to training_plots/
2. Rename folder to training_plots_A/
3. Train with config B → saves to training_plots/
4. Compare images side-by-side
5. See which config works better
```

---

## 🎓 Advanced Tips

### Monitor Specific Metrics

If you only care about certain metrics, you can focus on those plots:
- **Episode reward + episode length** = Overall performance
- **Explained variance + value loss** = Value function health
- **Approx KL** = Training stability

### Early Stopping

If you see:
- Episode length plateauing after 500k
- Explained variance not improving after 1M
- Approx KL consistently too high

→ **Stop training**, adjust config, restart fresh

### Checkpoint Comparison

After each checkpoint (200k steps):
1. Note the reward/length in plots
2. Run `visualize.py` to see behavior
3. Compare with previous checkpoint
4. Decide to continue or adjust

---

## 🎉 Benefits

**Before**: Had to read terminal logs to understand training
**After**: Visual feedback shows progress at a glance!

**Benefits:**
- ✅ Catch problems early (bad config, not improving)
- ✅ Know when to stop (plateaued, mastered task)
- ✅ Compare experiments easily
- ✅ Understand training dynamics better
- ✅ More engaging to watch! 🍿

**Enjoy your live training visualization! 📈**
