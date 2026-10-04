# ELab uploads and linked resources

The ELab tools upload existing results; they do not run training or detection. Run commands from the repository root. Uploads create or change remote records, so check the selected artifacts and destination before running them.

## Configuration

```bash
export ELAB_HOST_URL="https://your-elab-instance.example"
export ELAB_API_KEY="your-api-key"
export ELAB_VERIFY_SSL="true"
```

Keep API keys out of committed files. Install the ELab dependencies in the environment used for these commands. The YAML under `tools/elab/config/` is reference documentation, not an automatically loaded runtime configuration.

## Upload existing results

```bash
python tools/elab_cli.py simple upload-training --label "janus_training"
python tools/elab_cli.py simple upload-test --label "janus_test"
```

Both accept `--title-prefix`, `--category`, and `--team`. The current simple uploader uses template 24, category 5 and team 1 when category/team are omitted; confirm these installation-specific IDs are appropriate before uploading.

| Command | Included artifacts |
|---|---|
| `simple upload-training` | Existing `logs/`, `checkpoints/`, `models/` directories |
| `simple upload-test` | Existing `logs/`, `detection_results/`, plus `test_results_summary.yaml` if present |

Directories are archived as labelled `.tar.gz` files. This can include multiple runs, not just the most recent experiment. `lightning_logs/` is **not** included by the simple training uploader, and the composite summary is not automatically attached by `upload-test`. Inspect the returned artifact list and add missing material deliberately.

The root convenience commands `python elab.py upload-training` and `python elab.py upload-test` are also available. For optional flags, inspect the selected CLI's help:

```bash
python tools/elab_cli.py simple --help
python tools/elab_cli.py full upload-test-run --help
```

## Link experiments and database items

After uploading, use the returned experiment ID as the target:

```bash
python tools/elab_cli.py simple link-resources \
  --experiment-id <TEST_EXPERIMENT_ID> \
  --experiments <TRAINING_EXPERIMENT_ID> \
  --items <ITEM_ID>
```

Although the simple upload parsers accept `--experiments` and `--items`, their upload functions currently do not apply those links. Run `link-resources` separately and verify the result.

The full CLI also supports creating an experiment with links:

```bash
python tools/elab_cli.py full create-with-links \
  --title "Particle analysis" --body "Analysis of the selected run" \
  --template <TEMPLATE_ID> --experiments <EXPERIMENT_ID> --items <ITEM_ID>
```

Do not interpret an absent link in an API response as proof of success. Check the target record, accessible linked resources and any per-link errors in the response.

## Full test-run uploader and existing records

The full CLI calls the test upload command `upload-test-run`, not `upload-test`:

```bash
python tools/elab_cli.py full upload-test-run --label "janus_test" --no-update
```

Without `--no-update`, it can search for an existing record to update. `--update-experiment <ID>` explicitly selects an existing experiment. Use these modes only when updating that record is intended; inspect help for supported metadata and linking options. Template/tag behavior depends on the ELab installation; do not reuse historical tag settings blindly.

For item-body edits:

```bash
python tools/elab_cli.py simple patch-item --item-id <ITEM_ID> --body-file update.html
```

Download the current body before editing, patch from that live state, then verify the title, body and attachments after submission. Do not replace an existing record from an old local copy.

## Verify each operation

Check the process exit status and returned experiment/item ID. Open the remote record and confirm the title, expected attachments and links. A partial failure can leave a created record or some completed uploads; inspect that state before retrying to avoid duplicates. Missing resources and permission failures require correcting IDs or access, not repeated upload attempts.

See [tools/README.md](tools/README.md) for other data-processing tools.
