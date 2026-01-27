#!/bin/bash

# Example script showing how to use the new linked resources functionality in elab_cli.py
# This script demonstrates linking experiments and items to experiments

echo "=== Linked Resources Functionality Examples ==="
echo ""

# 1. Basic usage - upload test run with linked experiments
echo "1. Upload test run with linked experiments:"
echo "   python src/elab_cli.py upload-test-run \\"
echo "     --label \"linked_resources_test\" \\"
echo "     --title-prefix \"Linked Resources Test\" \\"
echo "     --experiments 155 176 177"
echo ""

# 2. Upload test run with linked items
echo "2. Upload test run with linked items:"
echo "   python src/elab_cli.py upload-test-run \\"
echo "     --label \"linked_items_test\" \\"
echo "     --title-prefix \"Linked Items Test\" \\"
echo "     --items 1270 61"
echo ""

# 3. Upload test run with both experiments and items linked
echo "3. Upload test run with both experiments and items linked:"
echo "   python src/elab_cli.py upload-test-run \\"
echo "     --label \"full_linking_test\" \\"
echo "     --title-prefix \"Full Linking Test\" \\"
echo "     --experiments 155 176 \\"
echo "     --items 1270"
echo ""

# 4. Link resources to existing experiment
echo "4. Link resources to existing experiment:"
echo "   python src/elab_cli.py link-resources \\"
echo "     --experiment-id 179 \\"
echo "     --experiments 155 176 \\"
echo "     --items 1270"
echo ""

# 5. Create new experiment with linked resources
echo "5. Create new experiment with linked resources:"
echo "   python src/elab_cli.py create-with-links \\"
echo "     --title \"LodeSTAR Analysis with References\" \\"
echo "     --body \"Analysis experiment linking to previous runs and data files\" \\"
echo "     --template 24 \\"
echo "     --category 5 \\"
echo "     --team 1 \\"
echo "     --experiments 155 176 \\"
echo "     --items 1270"
echo ""

# 6. Link only experiments
echo "6. Link only experiments:"
echo "   python src/elab_cli.py link-resources \\"
echo "     --experiment-id 179 \\"
echo "     --experiments 155 176"
echo ""

# 7. Link only items
echo "7. Link only items:"
echo "   python src/elab_cli.py link-resources \\"
echo "     --experiment-id 179 \\"
echo "     --items 1270 61"
echo ""

echo "=== Notes ==="
echo "- Experiment IDs: 155, 176, 177, 179 (from our previous tests)"
echo "- Item IDs: 1270, 61 (from the working example JSON)"
echo "- Template ID: 24 (LodeSTAR template)"
echo "- Category ID: 5 (Full Run)"
echo "- Team ID: 1 (Molecular Nanophotonics Group)"
echo ""
echo "=== What Gets Linked ==="
echo "- Experiments: Creates bidirectional links between experiments"
echo "- Items: Links database items (files, datasets) to experiments"
echo "- All links use 'duplicate' action for proper elab integration"
echo ""
echo "=== Output Format ==="
echo "The commands will output JSON showing:"
echo "- Experiment creation/upload results"
echo "- Linked resources status (success/errors)"
echo "- Any errors that occurred during linking"
echo ""
echo "=== Error Handling ==="
echo "- Partial failures are reported but don't stop the process"
echo "- Resource linking errors are shown as warnings"
echo "- Check the output for any failed links"
