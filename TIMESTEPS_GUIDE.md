# Training Timesteps Guide

## Current Test Results (1000 Timesteps)

### PPO Training
- **Timesteps**: 1,000
- **Episodes**: ~200
- **Average Reward**: ~190 per episode
- **Status**: ⚠️ **Under-trained** - Model is still learning basic patterns

### DQN Training
- **Timesteps**: 1,000
- **Episodes**: ~200
- **Average Reward**: ~186 per episode
- **Status**: ⚠️ **Under-trained** - Model is still exploring

## Recommended Timesteps for Meaningful Results

### Minimum for Basic Learning
**10,000 - 50,000 timesteps**
- Model learns basic patterns
- Can make reasonable station selections
- Good for initial testing and debugging
- **Time**: ~5-30 minutes

### Good Performance
**100,000 - 500,000 timesteps**
- Model learns effective strategies
- Consistent station placement patterns
- Good coverage and wait time optimization
- **Time**: ~1-5 hours
- **Recommended for**: Research, development, comparison

### Optimal Performance
**500,000 - 2,000,000 timesteps**
- Model converges to near-optimal solutions
- Stable, consistent performance
- Best coverage and wait time metrics
- **Time**: ~5-20 hours
- **Recommended for**: Production, final results, publications

## Why More Timesteps Are Needed

### Episode Length
- Each episode = 5 station placements
- 1,000 timesteps ≈ 200 episodes
- **Not enough** to learn complex placement strategies

### Learning Complexity
- 84 candidate locations = large action space
- Sequential decisions require learning dependencies
- Coverage and wait time optimization is complex
- Need many episodes to explore and exploit

### Convergence Indicators
Look for these signs of good training:
- ✅ **Stable rewards**: Reward variance decreases over time
- ✅ **Consistent selections**: Model picks similar good stations
- ✅ **High coverage**: >80% population coverage
- ✅ **Low wait time**: <5 minutes average wait
- ✅ **Good trip service**: >500k trips/day served

## Training Progress Indicators

### Early Stage (1k-10k timesteps)
- High exploration
- Random-looking selections
- High reward variance
- **Current status**: You are here

### Learning Stage (10k-100k timesteps)
- Patterns emerging
- Some consistency in selections
- Reward improving
- Coverage increasing

### Convergence Stage (100k-500k timesteps)
- Stable selections
- Consistent high rewards
- Good coverage (>70%)
- Low wait times

### Optimal Stage (500k+ timesteps)
- Near-optimal solutions
- Very stable performance
- Excellent metrics
- Ready for evaluation

## Quick Start Recommendations

### For Quick Testing
```bash
python -m agents.train_ppo --total_timesteps 10000
python -m agents.train_dqn --total_timesteps 10000
```
**Time**: ~10-15 minutes
**Use**: Initial testing, debugging

### For Good Results
```bash
python -m agents.train_ppo --total_timesteps 100000
python -m agents.train_dqn --total_timesteps 100000
```
**Time**: ~1-2 hours
**Use**: Development, comparison, research

### For Best Results
```bash
python -m agents.train_ppo --total_timesteps 500000
python -m agents.train_dqn --total_timesteps 500000
```
**Time**: ~5-10 hours
**Use**: Final results, publications, production

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./logs/
```
Then open http://localhost:6006

**Watch for**:
- `rollout/ep_rew_mean` - Should increase and stabilize
- `train/loss` - Should decrease
- `env/coverage_fraction` - Should increase
- `env/avg_wait` - Should decrease

### Training Metrics to Track
1. **Episode Reward**: Should increase over time
2. **Coverage**: Should reach >70% with 5 stations
3. **Wait Time**: Should decrease to <5 minutes
4. **Trips Served**: Should increase with more stations

## Expected Performance at Different Stages

| Timesteps | Coverage | Wait Time | Reward | Status |
|-----------|----------|-----------|--------|--------|
| 1,000     | ~0-20%   | 8-10 min  | 180-200| ⚠️ Learning |
| 10,000    | ~40-60%  | 6-8 min   | 300-400| ✅ Basic |
| 100,000   | ~70-85%  | 4-6 min   | 500-600| ✅ Good |
| 500,000   | ~85-95%  | 3-5 min   | 600-700| ✅ Excellent |
| 1,000,000+| ~90-100% | 2-4 min   | 700+   | ✅ Optimal |

## Tips for Faster Training

1. **Use GPU** (if available): 5-10x faster
2. **Parallel environments**: Train multiple episodes simultaneously
3. **Early stopping**: Stop when metrics plateau
4. **Hyperparameter tuning**: Optimize learning rate, batch size
5. **Curriculum learning**: Start with easier scenarios

## Summary

**For meaningful results, you need**:
- **Minimum**: 10,000 timesteps (~10 min)
- **Recommended**: 100,000 timesteps (~1-2 hours)
- **Optimal**: 500,000+ timesteps (~5-10 hours)

**Current 1,000 timesteps is just a test** - models are still in early learning phase and need much more training to make good decisions.

