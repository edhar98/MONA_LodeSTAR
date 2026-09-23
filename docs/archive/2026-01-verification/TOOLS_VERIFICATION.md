# Tools Verification Report

**Date:** 2026-01-27  
**Branch:** Tools & Automation  
**Scope:** tools/, elab.py, ELAB configs. No new features; correctness and alignment with Core and docs.

## 1. Commands / checks and results

| Check | Command / action | Result | Notes |
|-------|------------------|--------|--------|
| ELAB root help | `python elab.py` (no args) | ok | Prints usage and available commands (upload-training, upload-test, cli-full, cli-simple). No `--help` flag; usage shown when argc &lt; 2. |
| ELAB root upload-training | `python elab.py upload-training` | ok | Subprocess runs `tools/elab/scripts/upload_training.py`. Requires ELAB_HOST_URL, ELAB_API_KEY. No real upload performed. |
| ELAB root upload-test | `python elab.py upload-test` | ok | Subprocess runs `tools/elab/scripts/upload_test.py`. Same env requirements. |
| ELAB tools CLI help | `python tools/elab_cli.py --help` | ok | Shows subcommands `full`, `simple`. |
| ELAB tools simple help | `python tools/elab_cli.py simple --help` | ok | simple subparser has no options; subcommands live inside simple CLI. |
| ELAB simple upload-training | `python tools/elab_cli.py simple upload-training --help` | ok | After fix: argv forwarded as `sys.argv[2:]` to elab_cli_simple.main(); shows upload-training options. |
| ELAB config load | Inspect tools/elab and scripts | fail → documented | `tools/elab/config/elab_config.yaml` is **not** loaded by any ELAB script. elab_cli_simple and upload_training/upload_test use hardcoded defaults (template 24, category 5, team 1, title_prefix in parser). Documented as “active config” but only reference at present. |
| tdms_to_png invokable | `python tools/tdms_to_png.py --help` | ok | Has -o/--output, --list-structure, --to-mp4, --fps, --start-index, --num-files, --normed/--no-normed; matches tools/README.md. |
| crop invokable | `python tools/crop.py --help` | ok | Positionals: input, output. Matches tools/README. |
| mask invokable | `python tools/mask.py --help` | ok | Positionals: input, output. Matches tools/README. |
| merge_mp4 invokable | `python tools/merge_mp4.py --help` | ok | Positional input, -o/--output required, --start-index, --num-files, --fps, -f/--force. Matches tools/README. |
| wandb_logging from Core | Run `python src/train_single_particle.py --help` from repo root | ok | train_single_particle inserts `os.path.abspath(os.path.join(dirname(__file__), '..', 'tools'))` then `from wandb_logging import ...`. When run from repo root, tools path resolves to repo/tools; import succeeds per BASELINE_REPORT. |
| Web tdms_to_png import | Inspect web/app.py | ok | `sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))` then `from tdms_to_png import extract_images_from_tdms` (and list_tdms_structure, save_images, save_video). Matches QUICK_REFERENCE and INVENTORY. |

## 2. What was verified (code and docs only)

- **ELAB CLI:** elab.py (root) delegates to tools/elab/scripts and tools/elab/cli/*.py; tools/elab_cli.py exposes `full` and `simple` and, after fix, forwards `sys.argv[2:]` to the underlying CLIs. Subcommands of the simple CLI are `upload-training`, `upload-test`, `link-resources` (not “upload-training-results” / “upload-test-results” as in some doc examples).
- **ELAB env:** ELAB_HOST_URL, ELAB_API_KEY, ELAB_VERIFY_SSL are read in tools/elab/cli/elab_cli_simple.py and elab_cli.py via os.environ. Documented in docs/QUICK_REFERENCE.md (Environment Variables) and tools/README.md (ELAB Integration).
- **Data tools:** tdms_to_png, crop, mask, merge_mp4 are standalone scripts with argparse; usage/help matches tools/README.md. Paths are positional or -o; no doc’d requirement for a specific CWD for these tools.
- **Core integration:** train_single_particle adds repo-relative `tools` to sys.path and imports wandb_logging; web/app adds repo-relative tools and imports tdms_to_png. Both assume run/import from repo layout (run from repo root or uvicorn/web from repo root).

## 3. What was skipped and why

- **Real ELAB uploads:** Skipped. Only CLI startup and (for config) code paths were checked.
- **Running data tools on real files:** Skipped. Invocation and --help were verified from code; no file I/O was run.
- **Running Web or Core training:** Skipped. Imports and path logic were verified from code.

## 4. Recommended entry points for agents / automation

| Use case | Entry point | CWD | Env |
|----------|-------------|-----|-----|
| Upload training results | `python elab.py upload-training` or `python tools/elab/scripts/upload_training.py` | Repo root | ELAB_HOST_URL, ELAB_API_KEY, ELAB_VERIFY_SSL |
| Upload test results | `python elab.py upload-test` or `python tools/elab/scripts/upload_test.py` | Repo root | Same |
| ELAB “simple” CLI (upload-training, upload-test, link-resources) | `python tools/elab_cli.py simple <subcommand> [options]` | Repo root | Same |
| ELAB full CLI | `python tools/elab_cli.py full [subcommand] [options]` | Repo root | Same |
| TDMS → PNG/MP4, crop, mask, merge_mp4 | `python tools/<script> ...` | Any; paths are explicit or -o | None for these tools |

**Recommendation:** For automated/agent runs, use **`elab.py`** for upload-training/upload-test (one place, same CWD as Core) or **`tools/elab_cli.py simple <subcommand>`** if you need the unified simple CLI. Prefer **`tools/elab/scripts/upload_training.py`** / **`upload_test.py`** only when calling the scripts directly is required. Do **not** rely on `tools/elab/config/elab_config.yaml` being loaded by current code; treat it as reference until load logic is added.

## 5. Doc updates made

- **docs/QUICK_REFERENCE.md:** ELAB “Using direct CLI” examples use `upload-training` and `upload-test` (replacing upload-training-results / upload-test-results). ELAB env vars already include ELAB_VERIFY_SSL.
- **docs/BRANCH_GUIDES.md:** Tools & Automation “Upload Training/Test Results” direct-CLI examples use `upload-training` and `upload-test`. ELAB_VERIFY_SSL already in environment setup.
- **tools/README.md:** ELAB “Using direct CLI” examples use `upload-training` and `upload-test`. Added short “Tools verification” pointer to TOOLS_VERIFICATION.md and noted that elab_config.yaml is not loaded by current scripts.
- **DUPLICATES_DOCUMENTATION.md:** Clarified recommended entry point for agents (elab.py vs tools/elab_cli.py vs tools/elab/cli/elab_cli_simple.py) and that tools/elab/config/elab_config.yaml is reference-only until load logic exists.

## 6. Code change

- **tools/elab_cli.py:** `full_cli_main()` and `simple_cli_main()` are called with `sys.argv[2:]` so that `python tools/elab_cli.py simple upload-training [options]` and `python tools/elab_cli.py full <subcommand> [options]` work as intended.

## 7. Open questions / gaps

1. **ELAB config:** `tools/elab/config/elab_config.yaml` defines template, category, team, directories, title_prefixes, but no script reads it. Options: (a) add YAML load in upload_training/upload_test and/or elab_cli_simple, or (b) keep as reference and document that defaults are hardcoded.
2. **Subcommand naming:** Docs previously used “upload-training-results” / “upload-test-results”; actual subcommands are “upload-training” / “upload-test”. All doc references updated to match code.
3. **CWD for ELAB scripts:** elab.py does not change CWD; scripts run with caller’s CWD. Upload scripts use CWD-relative dirs (e.g. `logs`, `models`, `detection_results`). Align with BASELINE and QUICK_REFERENCE: run from repo root.
