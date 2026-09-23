# Web frontend validation — 2026-09-23

Target: existing per-user MONA JupyterHub. This report covers independent frontend/static proxy validation of this working checkout. It does not certify the installed `/home/mona/MONA_LodeSTAR` checkout or a live browser session. Runtime ASGI/real-frame smoke validation is owned by the separate correction agent; no research features were added by this review.

## Checks performed

- Node is installed at `/usr/bin/node`. Parsed the single inline script from `web/templates/index.html` with `vm.Script`: passed.
- Inspected HTML literal IDs and JavaScript literal `getElementById` references: no duplicate IDs. The sole lookup without a literal HTML ID, `field-tip`, is created dynamically by `applyFieldHints`; it is not a missing control.
- Executed the actual `API_BASE`, `apiUrl`, and `wsUrl` helper code in isolated Node VM contexts, without fetching anything. HTTPS uses `wss`; standard root, numeric proxy, prefixed numeric proxy, and named service paths preserve their prefixes.
- Read initialization/auth/health, model selection, crop/training, single/batch detection, tracking, ABP display, job polling, downloads, and WebSocket handlers against corresponding backend entry points. Form requests use JSON or multipart as appropriate; model IDs in detection queries are URI-encoded; download and training WebSocket paths use the proxy helpers.
- Confirmed `/health` reports tracking, analysis, crescent availability, deployment mode, GPU availability, and user storage location. The frontend currently uses its GPU fields but does not disable unavailable workflow panels. Actual health response validation belongs to the isolated API smoke.

| Browser pathname | Constructed health path | Result |
|---|---|---|
| `/` | `/health` | Pass |
| `/user/alice/proxy/8765/` | `/user/alice/proxy/8765/health` | Pass |
| `/prefix/user/alice/proxy/8765/` | `/prefix/user/alice/proxy/8765/health` | Pass |
| `/user/alice/mona-track/` | `/user/alice/mona-track/health` | Pass |
| `/user/alice/mona-track-1/` | `/user/alice/mona-track-1/health` | Pass |
| `/user/alice/labserver/mona-track/` | `/health` | Unsupported named-server shape in reviewed code |

WebSocket results preserve the same successful prefixes with `wss://hub.example`, as expected. The named-server case is a conditional release blocker: determine the actual Hub launch URL before assuming support. No live Hub request was made.

## Browser tooling available locally

No exposed browser automation tool was available. Python modules `playwright`, `selenium`, and `pyppeteer`, and Node modules `playwright`, `puppeteer`, `selenium-webdriver`, and `jsdom` were absent. `geckodriver`, Chromium, and Chrome were not found on PATH. `/usr/bin/firefox` links to a shell launcher with an actual executable at `/usr/lib/firefox/firefox`.

At initial inspection Firefox had not been launched; the subsequently authorized bounded native-browser check is recorded below. No browser dependencies were installed. A full browser-driven acceptance pass needs an available automation interface or a manual session through the actual Hub proxy; do not present Node parsing or ASGI tests as browser evidence.

## Findings handed to the correction agent

1. **Training restart handling:** frontend `pollTrainJob` and `updateTrainProgress`, and backend `ws_training`, did not treat `interrupted` as terminal. State restoration now generates that status, so an existing page could poll indefinitely with training controls disabled. Acceptance: interruption stops polling/WS progress and restores controls with a clear rerun message.
2. **Resumed training controls:** `resumeActiveJobs` hides the Start button, while `resetTrainUI` only enables it. Acceptance: completing, failing, cancelling, or interrupting a resumed job restores its visibility.
3. **ABP result truthfulness:** `showAbpResults` always displays “ABP fit complete”, ignores `plot_error`/`orientation_note`, and does not clear a prior plot when a new response has none. Acceptance: distinguish an available fit from insufficient data, show plot failure/orientation notes, and remove obsolete plot/download state before showing new results.

These findings concern existing workflow behavior, not requests to integrate experimental models. The first two were sent directly to the runtime correction agent and coordinator; ABP reporting was flagged alongside them. Correction status and final targeted evidence will be appended after re-review.

## Remaining release evidence

- Confirm the intended installed checkout/revision and actual Hub URL. This source checkout differs from the currently installed editable distribution.
- Through an isolated or approved real browser session, verify Hub identity initialization, file selection/upload, rendered image/crop interactions, saved-model selection, detection overlays, job updates, tracking/ABP output display, downloads, and restart recovery. No click-through or pixel-rendering validation has occurred here.
- Verify real TDMS/video decoding and browser playback where those features are in the release scope. Static JavaScript checks cannot establish codec availability.
- Preserve raw/scientific outputs; sparse-frame corrections do not automatically validate or regenerate historical physics results.

No live Hub state, installed checkout, external service, model, or dataset was modified by this reviewer. Repository edits were limited to this evidence document; the subsequently authorized local browser/HTTP checks used temporary storage only.

## Final correction review and bounded browser/HTTP check

The separate correction agent fixed interrupted training terminal handling in polling, UI, and both server WebSocket branches; Start becomes visible again after reset. ABP results now clear obsolete plots/downloads, distinguish no-fit responses, show escaped orientation notes and plot errors. Proxy prefix resolution now selects the last matching service segment, including a named server itself called `mona-track-lab`. Independently ran `node test/integration/check_web_ui.cjs` on final code: **passed complete JavaScript syntax, six proxy-prefix cases, interrupted training recovery, and ABP no-fit/plot-error/stale-image behavior**. Thus the named-server failure in the pre-correction table is resolved by the reviewed patch.

With explicit coordinator authorization, started current-checkout Uvicorn on `127.0.0.1:18765` with `MONA_TRACK_JUPYTER=1`, identity `review-user`, and isolated data/feedback/profile paths under `/tmp/mona-browser-review-A48g9F`. Initial sandbox socket access was denied; approved escalated commands were used for the local server/browser/client. No live Hub was contacted.

- Native Firefox headless completed successfully and produced `/tmp/mona-browser-review-A48g9F/mona-track.png` (49,598 bytes, 1440×1000). Visually inspected it: the initial MONA login shell renders correctly. The screenshot captures the initial state before asynchronous Hub identity completes, so it **does not establish automatic login or interactive workflow success**. Server logs show the browser fetched `/` and `/auth/me`, both HTTP 200. Initial missing-profile attempt was stopped, then a real temporary profile directory was used. Headless graphics warnings did not prevent the successful screenshot.
- Real HTTP `/health` returned 200, Jupyter mode, `review-user`, and tracking/analysis/crescent capabilities true. `/auth/me` returned the isolated identity and empty temporary session.
- Multipart-uploaded a harmless copy of `test/README.md` as `browser-check.csv` into temporary result storage (3,823 bytes). A real HTTP GET of `/results/review-user/download/browser-check.csv` returned 200 and matched the source SHA-256 exactly: `cbf30ff7fdba738fccf226ef1f9dac8beca8113035897ede31e1fcb61a209067`. This tests upload/download byte integrity, not CSV scientific validity. It demonstrates actual `FileResponse` streaming works despite the separate in-process ASGI harness's thread-related streaming stall.
- Firefox exited successfully. Stopped only the reviewer's own servers/browser attempts using their execution sessions; final Uvicorn shutdown completed cleanly. Temporary screenshot/profile/test artifacts remain available for inspection; no user data was deleted.

Coordinator final verification on stable code also passed **21 main tests**, the Node checks, and the separate real-model smoke: **367 detections across three real frames, 121 tracks/356 track rows, ABP plot, restart persistence, and 120 finite template-mode position/orientation/NCC rows on one frame**. Those real-model results are coordinator-reported evidence; this reviewer independently verified frontend behavior checks and actual local HTTP streaming above.

The reviewed bounded corrections are ready for scoped commit preparation. Remaining deployment gates are selecting/promoting the reviewed revision into the actual installed checkout and a real Hub-proxy browser acceptance session (identity completion, interactions, WebSocket progress and downloads). These checks neither deploy the app nor validate/regenerate historical physics outputs.
