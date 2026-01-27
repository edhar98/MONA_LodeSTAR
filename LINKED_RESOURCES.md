# Linked Resources Functionality

This document explains how to use the new linked resources functionality in `elab_cli.py` to create relationships between experiments and items in elab.

## Overview

The linked resources functionality allows you to:
- **Link experiments** to other experiments (creating bidirectional relationships)
- **Link items** (files, datasets) to experiments
- **Create experiments with pre-linked resources** in a single operation
- **Add links to existing experiments** after creation

## New Commands

### 1. `link-resources` - Link Resources to Existing Experiment

Links experiments and/or items to an existing experiment.

```bash
python src/elab_cli.py link-resources \
  --experiment-id <EXPERIMENT_ID> \
  --experiments <EXP_ID1> <EXP_ID2> ... \
  --items <ITEM_ID1> <ITEM_ID2> ...
```

**Arguments:**
- `--experiment-id`: ID of the target experiment to link resources to
- `--experiments`: List of experiment IDs to link (optional)
- `--items`: List of item IDs to link (optional)

**Example:**
```bash
python src/elab_cli.py link-resources \
  --experiment-id 179 \
  --experiments 155 176 \
  --items 1270
```

### 2. `create-with-links` - Create Experiment with Linked Resources

Creates a new experiment and immediately links resources to it.

```bash
python src/elab_cli.py create-with-links \
  --title "Experiment Title" \
  --body "Experiment description" \
  --template <TEMPLATE_ID> \
  --experiments <EXP_ID1> <EXP_ID2> ... \
  --items <ITEM_ID1> <ITEM_ID2> ...
```

**Arguments:**
- `--title`: Experiment title (required)
- `--body`: Experiment description (required)
- `--template`: Template ID (required)
- `--category`: Category ID (optional)
- `--tags`: Tag IDs (optional)
- `--team`: Team ID (optional)
- `--experiments`: List of experiment IDs to link (optional)
- `--items`: List of item IDs to link (optional)

**Example:**
```bash
python src/elab_cli.py create-with-links \
  --title "LodeSTAR Analysis with References" \
  --body "Analysis experiment linking to previous runs and data files" \
  --template 24 \
  --category 5 \
  --team 1 \
  --experiments 155 176 \
  --items 1270
```

### 3. Enhanced `upload-test-run` - Upload with Linked Resources

The existing `upload-test-run` command now supports linking resources during upload.

```bash
python src/elab_cli.py upload-test-run \
  --label "my_test_run" \
  --title-prefix "Test Run" \
  --experiments <EXP_ID1> <EXP_ID2> ... \
  --items <ITEM_ID1> <ITEM_ID2> ...
```

**New Arguments:**
- `--experiments`: List of experiment IDs to link
- `--items`: List of item IDs to link

**Example:**
```bash
python src/elab_cli.py upload-test-run \
  --label "linked_resources_test" \
  --title-prefix "Linked Resources Test" \
  --experiments 155 176 177 \
  --items 1270
```

## How Linking Works

### Experiment Links
- Creates **bidirectional relationships** between experiments
- Uses `action="duplicate"` for proper elab integration
- Links appear in both experiments' "Related Experiments" sections

### Item Links
- Links database items (files, datasets) to experiments
- Uses `action="duplicate"` for proper elab integration
- Items appear in the experiment's "Related Items" section

### Link Actions
All links use the `"duplicate"` action, which:
- Imports the links from the source resource
- Creates proper elab relationships
- Maintains data integrity

### **Important Note: Link Visibility**
- **Links are created successfully** via the API (returns 201 Created)
- **Links are stored internally** by elab
- **Links are displayed in the web interface** according to template configuration
- **Links may not appear in standard API response fields** (`experiments_links`, `related_experiments_links`, etc.) due to template configuration
- **This is normal behavior** - the linking functionality works correctly even if links aren't visible in API responses

## Output Format

### Successful Linking
```json
{
  "experiment_id": 179,
  "experiment_links": [
    {
      "id": 155,
      "status": "linked",
      "type": "experiment"
    },
    {
      "id": 176,
      "status": "linked",
      "type": "experiment"
    }
  ],
  "item_links": [
    {
      "id": 1270,
      "status": "linked",
      "type": "item"
    }
  ],
  "errors": []
}
```

### Partial Success with Errors
```json
{
  "experiment_id": 179,
  "experiment_links": [
    {
      "id": 155,
      "status": "linked",
      "type": "experiment"
    }
  ],
  "item_links": [],
  "errors": [
    "Failed to link experiment 176: (404) Not Found",
    "Failed to link item 1270: (403) Forbidden"
  ]
}
```

## Error Handling

### Partial Failures
- Individual link failures don't stop the entire process
- Successful links are reported
- Failed links are listed in the `errors` array
- Return codes indicate success level:
  - `0`: Complete success
  - `2`: Partial success with errors
  - `3`: Complete failure

### Common Error Scenarios
- **404 Not Found**: Resource ID doesn't exist
- **403 Forbidden**: Insufficient permissions
- **400 Bad Request**: Invalid parameters
- **500 Internal Server Error**: Server-side issues

## Use Cases

### 1. **Test Run Analysis**
Link test run experiments to:
- Previous test runs for comparison
- Reference datasets and files
- Analysis results from other experiments

### 2. **Data Pipeline Tracking**
Link experiments to:
- Input data files
- Intermediate processing steps
- Final output datasets

### 3. **Collaborative Research**
Link experiments to:
- Related work from team members
- Shared datasets and resources
- Published results and references

### 4. **Reproducibility**
Link experiments to:
- Source code repositories
- Configuration files
- Dependencies and requirements

## Best Practices

### 1. **Resource Identification**
- Use meaningful labels for experiments
- Document the purpose of each link
- Keep track of resource IDs in your workflow

### 2. **Link Management**
- Link resources during experiment creation when possible
- Use the `link-resources` command for post-creation linking
- Regularly review and update links as needed

### 3. **Error Handling**
- Check the output for any failed links
- Verify resource IDs exist before linking
- Ensure you have proper permissions for all resources

### 4. **Workflow Integration**
- Integrate linking into your automated workflows
- Use consistent naming conventions
- Document linking patterns for your team

## Examples

### Complete Workflow Example
```bash
# 1. Upload test run with linked resources
python src/elab_cli.py upload-test-run \
  --label "janus_particle_analysis" \
  --title-prefix "Janus Particle Analysis" \
  --experiments 155 176 \
  --items 1270

# 2. Create analysis experiment linking to the test run
python src/elab_cli.py create-with-links \
  --title "Janus Particle Detection Analysis" \
  --body "Analysis of Janus particle detection results" \
  --template 24 \
  --experiments 179 \
  --items 1270

# 3. Add additional links to existing experiment
python src/elab_cli.py link-resources \
  --experiment-id 180 \
  --experiments 155 \
  --items 61
```

### Batch Linking Example
```bash
# Link multiple experiments to a central analysis experiment
python src/elab_cli.py link-resources \
  --experiment-id 180 \
  --experiments 155 176 177 178 179
```

## Troubleshooting

### Common Issues

1. **"Failed to link experiment X: (404) Not Found"**
   - Verify the experiment ID exists
   - Check if you have access to the experiment

2. **"Failed to link item X: (403) Forbidden"**
   - Verify you have read permissions for the item
   - Check if the item is accessible to your team

3. **"Failed to link resource X: (500) Internal Server Error"**
   - Server-side issue, try again later
   - Check elab server status

4. **"Links created successfully but not visible in API response"**
   - **This is normal behavior** - links are created and stored by elab
   - **Check the web interface** - links should be visible there
   - **Template configuration** controls how links are displayed
   - **API response fields** may not show links due to template settings

### Debugging Tips

1. **Use the `resource-check` command** to verify resource existence:
   ```bash
   python src/elab_cli.py resource-check --id 155
   ```

2. **Check resource permissions** in the elab web interface

3. **Verify resource IDs** from the elab web interface or API responses

4. **Use debug mode** by setting `ELAB_VERIFY_SSL=false` if needed

5. **Verify link creation** by checking the web interface after running link commands

## Integration with Existing Workflows

The linked resources functionality integrates seamlessly with existing workflows:

- **No changes required** to existing commands
- **Backward compatible** with all previous functionality
- **Enhances** the `upload-test-run` command
- **Provides new commands** for advanced linking scenarios

## Future Enhancements

Potential future improvements could include:
- **Bulk linking** operations
- **Link templates** for common patterns
- **Link validation** and verification
- **Link visualization** and reporting
- **Automated linking** based on metadata patterns
