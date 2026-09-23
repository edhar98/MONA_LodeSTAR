# Deployment comparison — 2026-09-23

Read-only comparison of the reviewed source checkout and the installed JupyterHub source. No target files, services, Git configuration, index, or environment packages were changed. Runtime/user data and credentials were not read. This report is a deployment preflight, not deployment completion.

## Verified state

| Item | Evidence |
|---|---|
| Reviewed source | `/home/edgarharutyunyan/MONA_LodeSTAR`, branch `dev` |
| Installed target | `/home/mona/MONA_LodeSTAR`, `.git/HEAD` points to `refs/heads/main` |
| Shared base revision | Both source HEAD and target loose `main` ref are `7fb78e29c3b9f1591a6d2dd6f2d25d6614c2d039` |
| Target ownership/access | Directory and `.git` owned by `nobody:nogroup`, mode 0755; current uid 1043 cannot write target |
| Installed distribution outside checkout | Running `/opt/mona_jupyterhub_env/bin/python` from `/tmp` reports `mona-track` 0.2.0, editable URL `file:///home/mona/MONA_LodeSTAR`, `web` origin in that target |
| Launcher entry point | `jupyter_serverproxy_servers`: `mona-track = web.jupyter_config:setup_mona_track` |
| Packaging/launcher comparison | `setup.py`, `web/jupyter_config.py`, `web/jupyter_launch.py`, and `web/config.py` are byte-identical between source and target |
| Target runtime-file comparison | All source-HEAD tracked `.py`, `.yaml`, `.html`, `.svg`, `.txt` under `web/`, `src/`, `tools/`, `test/`, `debug/`, plus `setup.py`, exist at target and match the shared HEAD bytes |

No target AGENTS.md exists. Target CLAUDE.md is available and retains the existing repository workflow guidance, including committed-Core integration and `dev`/`main`. Its historical paths/progress differ from current corrected documentation.

Inside the source checkout, stale local egg-info instead reports 0.1.0 with no entry points. That CWD-dependent metadata result does not identify the Hub installation; use an outside-checkout inspection for deployment evidence.

## Remaining target cleanliness gate

Ordinary target `git status`/`log` fail with dubious ownership. The explicitly authorized read-only retry using `--no-optional-locks -c safe.directory=/home/mona/MONA_LodeSTAR -c core.fsmonitor=false` also failed. `readlink -f` confirms this is already the canonical target path; installed Git reports 2.25.1. No persistent safe-directory configuration was added. Thus target index, untracked files, and complete Git cleanliness are not certified; byte comparisons above cover only the stated runtime-file subset.

An authorized owner/admin context must run target status, staged/unstaged diff, and revision checks before promotion. Preserve any unrelated target modifications or untracked collisions; do not reset or clean them. No source commit containing this correction set existed when this comparison was made, so there is no release commit hash to promote yet. Shared ancestry makes a fast-forward plausible once reviewed commits exist, not yet proven safe.

## Promotion contents and provenance

At inspection the exact modified Python paths were:

- `src/tracking/track_particles.py`
- `src/analysis/analyze_tracks.py`
- `tools/elab_cli.py`
- `web/app.py`
- `web/routers/files.py`
- `web/routers/tdms_explorer.py`
- `web/state.py`

The follow-up promotion scope also includes `web/templates/index.html`, the new `test/unit/test_review_regressions.py`, `test/integration/smoke_hub_pipeline.py`, `test/integration/check_web_ui.cjs`, and updated `test/README.md`. Include the independently reviewed documentation/archive/ignore changes in their own commit scope. Existing source dirty documentation and TDMS work predate this task: `web/routers/tdms_explorer.py` was already modified at the initial inventory, so do not attribute or discard its entire diff as agent-owned. Root research scripts/notebooks/checkpoints, `Notes.txt`, ELab artifacts, and other unrelated untracked work are not automatic deployment inputs. Review explicit paths/hunks rather than staging everything.

Use the coordinator's final reviewed commit list as the authoritative release manifest; this report's path list is a snapshot while other agents may still be testing. Keep the Core commit available before Web integration, consistent with the repository gate.

Suggested logical commit groups, in order (not staged or committed by this report):

1. **Core elapsed-frame corrections:** `src/tracking/track_particles.py` and `src/analysis/analyze_tracks.py`.
2. **ELab dispatcher:** `tools/elab_cli.py`.
3. **Web integration and regression evidence:** the four modified web Python files above, `web/templates/index.html`, `test/unit/test_review_regressions.py`, both opt-in integration checks, and `test/README.md`. The combined regression file covers Core/ELab as well as Web and should validate all preceding commits together.
4. **Documentation/cleanup:** reviewed live Markdown updates, exact archive moves, new workflow/review/cleanup/deployment reports and narrow `.gitignore` rules. Explicitly review inherited documentation changes; exclude unrelated research artifacts.

The real-model smoke reported 3 frames/367 detections/121 tracks/356 rows, plus 120 finite template detections with a 10-degree angle step and 2-pixel search radius. This verifies bounded pipeline/schema behavior, not scientific fit or orientation accuracy. Native browser evidence is pending independent review; the Node check uses mocked elements. See [test instructions](../test/README.md) for fixture requirements and remaining HTTP-streaming/TDMS/browser limits.

## Deployment mechanism and rollback plan

1. Finish representative browser/pipeline checks and independent verification in the source checkout. Create scoped reviewed commits and record their exact hashes plus the base revision above.
2. In an authorized target owner/admin context, verify branch/ref, full status (including untracked collisions), and write access. If target work conflicts with the release, stop for reconciliation without resetting it.
3. Preserve an explicit rollback ref at the old target revision and record the release manifest. Transfer the exact reviewed commits through a Git fetch from the approved source or a verified bundle, then require `merge --ff-only` of the approved release hash. Do not use an indiscriminate file copy or floating remote branch as the release definition.
4. The existing editable source location and launcher entry point are unchanged, so no global reinstall is implied by these code-only changes. Verify outside-CWD import origin and actual revision after promotion. Packaging still requires adjacent `src/` and `tools/`; a standalone wheel is not established.
5. Coordinate stopping only the affected per-user MONA process after ensuring no jobs should continue; restart through the existing Hub launcher. The launcher binds loopback and sets identity plus `MONA_TRACK_HOME` (default per-user `~/mona_track`). State/data paths remain outside the code release; preserve them. Daemon jobs do not resume computation after restart.
6. Run a representative Hub workflow and inspect saved artifacts, then record the running revision. Health/import checks alone are insufficient.

Rollback is a deliberate owner/admin operation: stop the affected app, preserve current user state, and return the code to the recorded prior release using a separate checkout/release switch or reviewed revert commits. Do not reset a dirty target, delete user outputs, or overwrite newly generated model/state files. New job-specific model files and corrected physics results are data artifacts, not files to erase when reverting code. Verify prior-code compatibility with preserved state before reopening the launcher.

The concrete blockers for executing promotion are target Git cleanliness/ownership access and an approved final commit manifest. Browser verification and restart coordination remain operational gates. Historical physics outputs must be regenerated deliberately with provenance; promoting code does not update old reports or authorize ELab changes.

## Source commit preparation update

The coordinator reports three reviewed source commits on `dev`: Core corrections `fb3fd9b`, ELab forwarding `28a53d5`, and Web/tests `84da4bc`. The documentation/cleanup commit is pending. These source commits supersede the earlier uncommitted-source snapshot above; target deployment is unchanged. Record the final documentation commit and complete the target cleanliness, ownership, and operational gates before promotion.
