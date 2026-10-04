# JupyterHub deployment

The supported deployment runs one MONA Track backend per Jupyter user through
the JupyterLab launcher. JupyterHub provides authentication; no second app
password is required in this mode.

## Release status

The tested web code is committed on `dev` as `1fbc661`. On 2026-10-04, 74 tests,
three frontend checks, and both real-model API smoke tests passed. The user
reported the nine manual web acceptance checks passed. These checks do not
replace post-update testing through the actual Hub launcher.

The installed checkout is `/home/mona/MONA_LodeSTAR`, separate from the development
checkout. At the last read-only check it was on `main` at `7fb78e29`, owned by
`nobody:nogroup`. Target Git cleanliness could not be verified from the development
account because Git rejected its ownership. No live update or restart was made.
The administrator must verify the target state before applying a release.

## Update the installed checkout

Perform the update from an account authorized to manage the installed checkout.

1. Check `git status --short`, the current branch, and `git rev-parse HEAD`.
   Preserve local changes and investigate untracked collisions. Do not use
   reset or clean commands to force an update.
2. Record a rollback reference for the old revision. Fetch the approved release,
   inspect the commit range, and merge the exact release commit with
   `git merge --ff-only RELEASE_COMMIT`. Replace the placeholder with the approved
   hash; do not deploy an unreviewed branch tip.
3. Keep the full checkout, including adjacent `src/` and `tools/`. The installed
   editable package uses that layout. These code-only updates do not by
   themselves require reinstalling packages; verify the package import location
   before deciding to reinstall.
4. Save notebooks and finish or cancel jobs, then restart the affected user's
   Jupyter server. The parent process must reload the launcher configuration;
   restarting only the MONA child process is insufficient for the new private
   proxy header. Avoid restarting other users or the whole Hub unnecessarily.
5. Open MONA Track from the JupyterLab launcher and perform the checks below.

The package entry point is `web.jupyter_config:setup_mona_track` in the
`jupyter_serverproxy_servers` group. The launcher binds the backend to loopback
and sets the current Hub identity and data directory.

## Authentication and storage

The launcher generates a private capability and supplies it to the backend
environment and proxy header override. Hub-mode HTTP and WebSocket requests
without the matching header are rejected. Never place this value in a browser
URL, documentation, or logs. OS permissions must isolate users' home directories,
process environments, and authorized shared directories.

Hub user storage defaults to `~/mona_track/`. Standalone development storage
defaults to `web/data/USERNAME/`; these are different locations. Do not silently
copy test accounts, sessions, or files into Hub storage. Any migration needs an
explicit user mapping and backup.

A code update does not transfer ignored model weights. The ordinary tracker
works without an LSTM checkpoint. Optional gap checkpoints require separate,
approved provisioning and remain experimental.

## Verify after updating

- The launcher opens without a second login and reports the correct Hub user.
- Uploads, server-path reads, and previews work.
- Training progress works through the WebSocket or polling fallback.
- Requests for another user's resources are rejected.
- Direct backend HTTP without the private header returns 403.
- A small detection, tracking, visualization, and export workflow succeeds.
- Repeated output names preserve older results, and user data remains editable.

Run the [automated checks](../test/README.md) in the same environment where
practical. Record the installed release hash after validation.

## Development and connection troubleshooting

For a local-only development server:

```bash
python -m uvicorn web.app:app --host 127.0.0.1 --port 8002
```

To access it through an existing Jupyter user server, try
`/user/YOUR_USERNAME/proxy/8002/` on the Hub origin. Keep the final slash.
This requires the proxy extension and network reachability from that user
server to the running app. It does not enable automatic Hub identity in a
manually started standalone process; that app retains its separate test login.

For explicitly approved trusted-network testing, `--host 0.0.0.0` listens on all
interfaces. Standalone mode is not secure multi-user authentication. A reachable
Hub port does not imply the VPN permits direct access to port 8002.

- Connection refused: check the listener and address with `ss -ltnp`.
- Connection timeout: check VPN routing, firewall policy, and the return route.
- HTTP 403 in Hub mode: verify the parent launcher configuration was reloaded.
- Missing models: the web catalog contains user-trained models only.
- Missing files after switching modes: check the storage directory before moving data.

## Rollback

Stop only the affected service after coordinating active work. Preserve current
user data and outputs. Restore the recorded code revision using a separate
release checkout or reviewed revert; never reset a dirty installation or delete
new results. Check older-code compatibility with saved state before reopening
the launcher.

See [known limitations](KNOWN_ISSUES.md) before broader deployment.
