# Fix Distributed Training Warnings in Deeplay Application

## 🐛 Problem Description

When using Deeplay's `Application` class (which LodeSTAR inherits from) in distributed training scenarios, PyTorch Lightning generates warnings about missing `sync_dist=True` parameter in metric logging calls.

### Warning Messages:
```
/opt/mona_jupyterhub_env/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/logger_connector/result.py:434: 
It is recommended to use `self.log('train_between_image_disagreement', ..., sync_dist=True)` 
when logging on epoch level in distributed setting to accumulate the metric across devices.

/opt/mona_jupyterhub_env/lib/python3.10/site-packages/lightning/pytorch/trainer/connectors/logger_connector/result.py:434: 
It is recommended to use `self.log('train_within_image_disagreement', ..., sync_dist=True)` 
when logging on epoch level in distributed setting to accumulate the metric across devices.
```

## 🔧 Solution

Add `sync_dist=True` parameter to all `self.log()` calls in the `Application` class methods to ensure proper metric synchronization across devices in distributed training.

## 📍 Files to Modify

**File:** `/opt/mona_jupyterhub_env/lib/python3.10/site-packages/deeplay/applications/application.py`

## 🔄 Required Changes

### 1. Fix `training_step` method (around line 252-261)

**Before:**
```python
for name, v in loss.items():
    self.log(
        f"train_{name}",
        v,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
```

**After:**
```python
for name, v in loss.items():
    self.log(
        f"train_{name}",
        v,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        sync_dist=True,  # ✅ ADD THIS LINE
    )
```

### 2. Fix `validation_step` method (around line 276-285)

**Before:**
```python
for name, v in loss.items():
    self.log(
        f"val_{name}",
        v,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
```

**After:**
```python
for name, v in loss.items():
    self.log(
        f"val_{name}",
        v,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        sync_dist=True,  # ✅ ADD THIS LINE
    )
```

### 3. Fix `test_step` method (around line 304-313)

**Before:**
```python
for name, v in loss.items():
    self.log(
        f"test_{name}",
        v,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
    )
```

**After:**
```python
for name, v in loss.items():
    self.log(
        f"test_{name}",
        v,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        sync_dist=True,  # ✅ ADD THIS LINE
    )
```

### 4. Fix `log_metrics` method (around line 340-346)

**Before:**
```python
for name, metric in metrics.items():
    self.log(
        name,
        metric,
        **logger_kwargs,
    )
```

**After:**
```python
for name, metric in metrics.items():
    self.log(
        name,
        metric,
        sync_dist=True,  # ✅ ADD THIS LINE
        **logger_kwargs,
    )
```

## 🎯 Benefits

1. **Eliminates Warning Messages** - No more PyTorch Lightning distributed training warnings
2. **Proper Metric Synchronization** - Metrics are correctly accumulated across all devices
3. **Better Distributed Training** - Ensures consistent behavior in multi-GPU setups
4. **Follows Best Practices** - Aligns with PyTorch Lightning recommendations

## 🧪 Testing

After applying these changes, distributed training should run without warnings:

```bash
# Before fix - shows warnings
python train_model.py --devices 2

# After fix - clean output
python train_model.py --devices 2
```

## 📝 Commit Message

```
fix: add sync_dist=True to Application logging for distributed training

- Add sync_dist=True parameter to all self.log() calls in training_step, validation_step, test_step, and log_metrics methods
- Fixes PyTorch Lightning warnings about missing sync_dist parameter in distributed training
- Ensures proper metric synchronization across devices in multi-GPU setups
- Follows PyTorch Lightning best practices for distributed training

Fixes: #XXX (if applicable)
```

## 🔗 Related Issues

- PyTorch Lightning distributed training warnings
- LodeSTAR training with multiple devices
- Metric synchronization in distributed setups

---

**Status:** ✅ Changes Applied  
**Tested:** ✅ Verified with LodeSTAR training  
**Impact:** Low risk, backward compatible
