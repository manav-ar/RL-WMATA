# Fixes Applied for Training Issues

## ✅ Fixes Implemented

### 1. **Fixed GeoJSON to CSV Conversion** (`simulator/env.py`)
- **Problem**: Code assumed `lat`/`lon` columns exist, but GeoJSON stores coordinates in `geometry`
- **Fix**: Now extracts `lat`/`lon` from `geometry.x` and `geometry.y` if columns don't exist
- **Impact**: Ensures correct coordinates are loaded

### 2. **Added Data Validation** (`simulator/env.py`)
- **Problem**: No validation of candidate coordinates
- **Fix**: Added checks for:
  - Duplicate coordinates (warns if < 50% unique)
  - Invalid coordinates (outside DC area: lat 38.5-39.2, lon -77.2 to -76.8)
  - Sample coordinate display for debugging
- **Impact**: Catches data issues early

### 3. **Fixed CRS Warning** (`scripts/enrich_candidates_with_existing_stations.py`)
- **Problem**: Distance calculations in geographic CRS (EPSG:4326) cause warnings
- **Fix**: 
  - Projects to UTM Zone 18N (EPSG:32618) for distance calculations
  - Converts back to EPSG:4326 for saving
  - Suppresses warnings with proper handling
- **Impact**: No more CRS warnings, accurate distances

### 4. **Added Coverage Calculation Debugging** (`simulator/env.py`)
- **Problem**: Coverage showing 0% even with stations placed
- **Fix**: Added debug output showing:
  - Placed indices
  - Distance ranges
  - Coverage mask statistics
  - Final coverage calculation
- **Impact**: Helps identify why coverage is 0%

### 5. **Improved Enrichment Script** (`scripts/enrich_candidates_with_existing_stations.py`)
- **Problem**: CSV export might miss lat/lon
- **Fix**: Ensures lat/lon are extracted before CSV export
- **Impact**: Consistent data format

## 🔍 Diagnostic Tools Added

### `check_candidates_data.py`
Run this to diagnose candidate data issues:
```bash
python check_candidates_data.py
```

This will show:
- Coordinate ranges
- Duplicate coordinate counts
- Sample data
- Validation warnings

## 🐛 Known Issues to Investigate

### Issue 1: All Candidates Have Same Coordinates
**Symptom**: All selected stations show same lat/lon (38.895037, -77.036543)

**Possible Causes**:
1. CSV file has duplicate coordinates (data issue)
2. GeoJSON conversion didn't extract coordinates correctly
3. Candidates data itself has issues

**Solution**:
1. Run `python check_candidates_data.py` to diagnose
2. Check if CSV needs to be regenerated
3. Verify GeoJSON has correct geometry

### Issue 2: Coverage is 0%
**Symptom**: Coverage shows 0.000 even with 5 stations placed

**Possible Causes**:
1. All candidates at same location (distance matrix all zeros)
2. Existing stations already cover everything (subtraction issue)
3. Distance matrix not computed correctly
4. Coverage calculation bug

**Solution**:
1. Check debug output from coverage calculation
2. Verify distance matrix is computed correctly
3. Check if `has_existing_stations` is causing subtraction issue
4. Verify catchment radius (800m) is reasonable

## 🔧 Next Steps

1. **Run diagnostic**:
   ```bash
   python check_candidates_data.py
   ```

2. **Regenerate CSV if needed**:
   - Delete `data/prepared/candidates_final.csv`
   - Restart training (will auto-convert)

3. **Check coverage debug output**:
   - Look for "DEBUG Coverage:" messages during training/testing
   - Verify distance ranges are reasonable
   - Check if coverage_mask has any True values

4. **Verify candidates data**:
   - Ensure GeoJSON has unique coordinates
   - Check if population data is correct
   - Verify candidate_id mapping

## 📊 Expected Behavior After Fixes

- ✅ No CRS warnings
- ✅ Correct lat/lon extraction
- ✅ Data validation warnings if issues found
- ✅ Debug output for coverage calculation
- ✅ Accurate distance calculations

## ⚠️ If Issues Persist

1. **Check candidates CSV directly**:
   ```python
   import pandas as pd
   df = pd.read_csv("data/prepared/candidates_final.csv")
   print(df[['candidate_id', 'lat', 'lon']].head(20))
   print(f"Unique coords: {df[['lat', 'lon']].drop_duplicates().shape[0]}")
   ```

2. **Test coverage calculation manually**:
   ```python
   from simulator.env import StationPlacementEnv
   env = StationPlacementEnv(...)
   obs, _ = env.reset()
   # Place a station
   obs, reward, done, trunc, info = env.step(0)
   print(f"Coverage: {info.get('coverage_fraction', 0)}")
   ```

3. **Check distance matrix**:
   ```python
   print(f"Distance matrix shape: {env.distance_matrix.shape}")
   print(f"Sample distances: {env.distance_matrix[0, :5]}")
   ```

## Summary

✅ **Fixed**: GeoJSON conversion, CRS warnings, data validation
🔍 **Added**: Debugging output, diagnostic script
⚠️ **Investigate**: Why all candidates have same coordinates, why coverage is 0%

The fixes should resolve the CRS warnings and improve data handling. The duplicate coordinates and 0% coverage issues need further investigation with the diagnostic script.

