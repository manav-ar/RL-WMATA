# Data Issue Explanation

## 🔍 The Problem

**Symptom**: All 5 selected stations show the same coordinates (38.895037, -77.036543) even though they have different candidate IDs (C0, C3, C1, C2, C4).

## 🎯 Root Cause

This is a **DATA issue**, not a training issue. Here's what's happening:

1. **Model selects different candidate IDs**: C0, C3, C1, C2, C4 (different actions)
2. **Test script looks up coordinates**: `candidates_df[candidates_df['candidate_id'] == selected_id]`
3. **All candidates have same coordinates in CSV**: So all lookups return the same lat/lon

## ❌ Will More Training Help?

**NO** - More training (even 1 million timesteps) will NOT fix this if:
- The CSV file has duplicate coordinates for different candidate_ids
- The GeoJSON has duplicate coordinates
- The data conversion didn't extract coordinates correctly

**The model is working correctly** - it's selecting different candidate IDs. The problem is the data file has duplicate coordinates.

## ✅ How to Fix

### Step 1: Check Your Data
```bash
python check_candidates_data.py
```

This will show:
- How many unique coordinates you have
- If there are duplicates
- Sample coordinate data

### Step 2: Fix the Data
```bash
python fix_candidates_data.py
```

This will:
- Regenerate CSV from GeoJSON
- Extract coordinates from geometry (source of truth)
- Validate coordinate uniqueness
- Save corrected CSV

### Step 3: Verify
```bash
python check_candidates_data.py
```

Should now show:
- ✅ Many unique coordinates
- ✅ Valid coordinate ranges
- ✅ No duplicate warnings

### Step 4: Retrain
```bash
# Retrain with fixed data
python -m agents.train_ppo --total_timesteps 100000
python -m agents.train_dqn --total_timesteps 100000
```

## 🔬 How to Verify It's Fixed

After fixing and retraining, test again:
```bash
python test_PPO.py
python test_DQN.py
```

**Expected output** (if fixed):
```
1. C0: Lat=38.927786, Lon=-77.110532, Population=5979
2. C3: Lat=38.895123, Lon=-77.036789, Population=5144
3. C1: Lat=38.901234, Lon=-77.045678, Population=5568
...
```

**Bad output** (current issue):
```
1. C0: Lat=38.895037, Lon=-77.036543, Population=5979
2. C3: Lat=38.895037, Lon=-77.036543, Population=5144  # Same coords!
3. C1: Lat=38.895037, Lon=-77.036543, Population=5568  # Same coords!
...
```

## 📊 Why This Happens

### Possible Causes:

1. **GeoJSON has duplicate coordinates**
   - Multiple candidates at same location
   - Data collection issue

2. **CSV conversion bug** (now fixed)
   - Old code didn't extract from geometry
   - All candidates got default/duplicate values

3. **Data processing issue**
   - Candidates were aggregated/clustered incorrectly
   - Lost coordinate diversity

## 🎯 Impact on Training

### If Data Has Duplicates:
- ✅ Model can still learn (selects different IDs)
- ❌ But all selections appear at same location
- ❌ Coverage calculation may be wrong (all stations at same spot)
- ❌ Reward signal is misleading
- ❌ Model can't learn spatial diversity

### If Data is Fixed:
- ✅ Model learns to select diverse locations
- ✅ Coverage calculation is accurate
- ✅ Reward signal is meaningful
- ✅ Model learns optimal spatial distribution

## 🚨 Critical: Fix Before Long Training

**Do NOT train for 1 million timesteps with bad data!**

1. Fix the data first (5 minutes)
2. Then train (hours)
3. Otherwise you waste time training on bad data

## Summary

- **Issue**: Data has duplicate coordinates
- **Solution**: Regenerate CSV from GeoJSON geometry
- **Training**: Won't help if data is wrong
- **Action**: Run `python fix_candidates_data.py` first

