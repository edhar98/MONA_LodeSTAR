# Duplicate Files Documentation

This document clarifies which files to use when duplicates exist in the codebase.

## ELAB CLI Entry Points

### Files
- `elab.py` (root directory)
- `tools/elab_cli.py` (tools directory)

### Purpose
Both files provide access to ELAB CLI tools, but serve different use cases:

**`elab.py` (root):**
- Convenience wrapper script for running ELAB tools from project root
- Uses subprocess to call tools in `tools/elab/` directory
- Provides simple command mapping: `upload-training`, `upload-test`, `cli-full`, `cli-simple`
- Usage: `python elab.py upload-training`

**`tools/elab_cli.py`:**
- Main ELAB CLI entry point within tools package
- Direct Python import and execution
- Provides subcommands: `full`, `simple`
- Usage: `python tools/elab_cli.py full` or `python tools/elab_cli.py simple`

### Recommendation
- **For root-level convenience:** Use `elab.py` (root)
- **For tools package integration:** Use `tools/elab_cli.py`
- **For direct script access:** Use `tools/elab/scripts/upload_training.py` or `tools/elab/scripts/upload_test.py`
- **For automated/agent runs:** Prefer `elab.py upload-training` / `elab.py upload-test` (CWD = repo root, same as Core), or `python tools/elab_cli.py simple upload-training` / `upload-test` when using the unified simple CLI. See TOOLS_VERIFICATION.md.

### Status
Both files are maintained. `elab.py` calls scripts under `tools/elab/` (scripts and cli); `tools/elab_cli.py` exposes the same logic via subcommands `full` and `simple`.

---

## ELAB Configuration Files

### Files
- `elab_config.yaml` (root directory)
- `tools/elab/config/elab_config.yaml` (tools/elab/config directory)

### Purpose

**`elab_config.yaml` (root):**
- **Type:** Reference/documentation file
- **Purpose:** Documents ELAB configuration structure, template IDs, category/team IDs, and example usage
- **Contains:** Template information, example commands, reference values
- **Status:** Documentation/reference only

**`tools/elab/config/elab_config.yaml`:**
- **Type:** Reference configuration (structure and defaults)
- **Purpose:** Documents ELAB configuration structure; actual defaults are hardcoded in scripts
- **Contains:** Default experiment settings, tags, directory mappings, file patterns, archive settings
- **Status:** Not loaded by current ELAB scripts; use as reference until load logic is added. See TOOLS_VERIFICATION.md.

### Recommendation
- **For reference/documentation:** See `elab_config.yaml` (root)
- **For actual configuration:** Edit `tools/elab/config/elab_config.yaml`
- **For environment-specific configs:** Create additional configs in `tools/elab/config/` or use environment variables

### Status
Both files serve different purposes:
- Root version = Documentation/Reference
- Tools version = Active Configuration

---

## Summary

| Duplicate Pair | Root File | Tools File | Recommendation |
|----------------|-----------|------------|----------------|
| ELAB CLI | `elab.py` (wrapper) | `tools/elab_cli.py` (direct) | Use root for convenience, tools for integration |
| ELAB Config | `elab_config.yaml` (reference) | `tools/elab/config/elab_config.yaml` (active) | Root for docs, tools for actual config |

Both duplicates are intentional and serve complementary purposes. No consolidation needed at this time.
