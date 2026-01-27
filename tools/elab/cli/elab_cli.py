import argparse
import json
import os
import sys
import tarfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import elabapi_python
import urllib3
import tempfile
import urllib.parse
import requests


def build_api_client() -> elabapi_python.ApiClient:
    host_url: Optional[str] = os.environ.get("ELAB_HOST_URL")
    api_key: Optional[str] = os.environ.get("ELAB_API_KEY")
    verify_ssl_env: str = os.environ.get("ELAB_VERIFY_SSL", "true").strip().lower()
    if not host_url or not api_key:
        raise RuntimeError("Missing ELAB_HOST_URL or ELAB_API_KEY in environment")
    
    # Fix host URL - remove trailing slash to avoid double slashes in API calls
    if host_url.endswith('/'):
        host_url = host_url.rstrip('/')
    
    verify_ssl_flag: bool = verify_ssl_env in ("1", "true", "yes")
    configuration: elabapi_python.Configuration = elabapi_python.Configuration()
    configuration.api_key["api_key"] = api_key
    configuration.api_key_prefix["api_key"] = "Authorization"
    configuration.host = host_url
    configuration.debug = False
    configuration.verify_ssl = verify_ssl_flag
    api_client: elabapi_python.ApiClient = elabapi_python.ApiClient(configuration)
    api_client.set_default_header(header_name="Authorization", header_value=api_key)
    return api_client


def archive_directories(directories: List[str], run_label: str) -> List[str]:
    archives: List[str] = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        archive_name: str = f"{run_label}_{os.path.basename(directory)}.tar.gz"
        with tarfile.open(archive_name, "w:gz") as tar:
            tar.add(directory, arcname=os.path.basename(directory))
        archives.append(archive_name)
    return archives


def print_json(data: Any) -> None:
    try:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(str(data))


def _extract_metadata_dict(item_dict: Dict[str, Any]) -> Dict[str, Any]:
    meta: Any = item_dict.get("metadata")
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    if isinstance(meta, dict):
        return meta
    return {}


def _fetch_item_raw(host_url: str, api_key: str, verify_ssl: bool, resource_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    headers = {"Authorization": api_key, "Accept": "application/json"}
    url: str = host_url.rstrip("/") + f"/items/{resource_id}"
    http: urllib3.PoolManager
    if verify_ssl:
        http = urllib3.PoolManager()
    else:
        http = urllib3.PoolManager(cert_reqs="CERT_NONE")
    try:
        resp = http.request("GET", url, headers=headers)
        raw_text: str = resp.data.decode("utf-8")
        return json.loads(raw_text), None
    except Exception as exc:
        return None, str(exc)


def _fetch_experiment_raw(host_url: str, api_key: str, verify_ssl: bool, experiment_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    headers = {"Authorization": api_key, "Accept": "application/json"}
    url: str = host_url.rstrip("/") + f"/experiments/{experiment_id}"
    http: urllib3.PoolManager
    if verify_ssl:
        http = urllib3.PoolManager()
    else:
        http = urllib3.PoolManager(cert_reqs="CERT_NONE")
    try:
        resp = http.request("GET", url, headers=headers)
        raw_text: str = resp.data.decode("utf-8")
        return json.loads(raw_text), None
    except Exception as exc:
        return None, str(exc)


def cmd_resource_check(resource_id: int, expected_metadata_keys: Optional[List[str]]) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    items_api: elabapi_python.ItemsApi = elabapi_python.ItemsApi(api_client)
    try:
        item: Any = items_api.get_item(resource_id)
    except Exception as exc:
        print(f"Failed to fetch resource {resource_id}: {exc}", file=sys.stderr)
        return 2
    item_dict: Dict[str, Any]
    try:
        item_dict = item.to_dict()  # type: ignore[attr-defined]
    except Exception:
        try:
            item_dict = json.loads(json.dumps(item, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            print(str(item))
            item_dict = {"raw": str(item)}

    configuration = api_client.configuration  # type: ignore[attr-defined]
    raw_item, raw_err = _fetch_item_raw(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        resource_id=resource_id,
    )

    output: Dict[str, Any] = {"resource": item_dict}
    if raw_item is not None:
        output["resource_raw"] = raw_item
    if raw_err is not None:
        output["raw_fetch_error"] = raw_err
    print_json(output)

    if expected_metadata_keys:
        # Prefer raw_item for metadata, else fallback to parsed item_dict
        holder: Dict[str, Any] = raw_item if isinstance(raw_item, dict) else item_dict
        meta_dict: Dict[str, Any] = _extract_metadata_dict(holder)
        # Also consider decoded block if provided by server
        if not meta_dict and isinstance(holder.get("metadata_decoded"), dict):
            meta_dict = holder["metadata_decoded"]
        search_scopes: List[Dict[str, Any]] = [meta_dict]
        if isinstance(meta_dict.get("extra_fields"), dict):
            search_scopes.append(meta_dict["extra_fields"])  # type: ignore[index]
        missing: List[str] = []
        for key in expected_metadata_keys:
            found: bool = any(isinstance(scope, dict) and key in scope for scope in search_scopes)
            if not found:
                missing.append(key)
        if missing:
            print_json({
                "metadata_check": "missing_keys",
                "missing": missing,
                "scopes": [list(scope.keys()) for scope in search_scopes if isinstance(scope, dict)],
            })
            return 3
        print_json({"metadata_check": "ok"})
    return 0


def cmd_experiment_check(experiment_id: int) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
    try:
        exp: Any = experiments_api.get_experiment(experiment_id)
    except Exception as exc:
        print(f"Failed to fetch experiment {experiment_id}: {exc}", file=sys.stderr)
        return 2
    exp_dict: Dict[str, Any]
    try:
        exp_dict = exp.to_dict()  # type: ignore[attr-defined]
    except Exception:
        try:
            exp_dict = json.loads(json.dumps(exp, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            print(str(exp))
            exp_dict = {"raw": str(exp)}

    configuration = api_client.configuration  # type: ignore[attr-defined]
    raw_exp, raw_err = _fetch_experiment_raw(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=experiment_id,
    )

    output: Dict[str, Any] = {"experiment": exp_dict}
    if raw_exp is not None:
        output["experiment_raw"] = raw_exp
    if raw_err is not None:
        output["raw_fetch_error"] = raw_err
    print_json(output)
    return 0


def cmd_upload_run(run_label: Optional[str], directories: List[str], title_prefix: str) -> int:
    label: str = run_label or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    archives: List[str] = archive_directories(directories, label)
    if not archives:
        print("No directories found to archive", file=sys.stderr)
        return 4
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
    uploads_api: elabapi_python.UploadsApi = elabapi_python.UploadsApi(api_client)
    title: str = f"{title_prefix} {label}".strip()
    body: str = "Automated upload of run artifacts: " + ", ".join(directories)
    try:
        exp_resp: Any = experiments_api.post_experiment_with_http_info(body={"title": title, "body": body})
    except Exception as exc:
        print(f"Failed to create experiment: {exc}", file=sys.stderr)
        return 5
    try:
        location_header: Optional[str] = exp_resp[2].get("Location")  # type: ignore[index]
        experiment_id: int = int(str(location_header).split("/").pop()) if location_header else int(exp_resp[0].id)  # type: ignore[attr-defined]
    except Exception:
        print_json({"unexpected_experiment_response": str(exp_resp)})
        return 6
    for archive in archives:
        try:
            with open(archive, "rb") as fh:
                uploads_api.post_upload(id=experiment_id, file=fh, filename=os.path.basename(archive))
        except Exception as exc:
            print(f"Failed to upload {archive} to experiment {experiment_id}: {exc}", file=sys.stderr)
            return 7
    print_json({"experiment_id": experiment_id, "uploaded": archives})
    return 0


def find_existing_experiment(experiments_api, title: str, template: int = 24) -> Optional[int]:
    """Find existing experiment by title and template"""
    try:
        # Try to get experiments with the same template
        # Note: This is a simplified approach - elabapi might not support direct search by title
        # We'll use a workaround by checking if we can get the experiment by ID if it's in the title
        return None
    except Exception:
        return None

def find_experiment_by_title_pattern(experiments_api, title_pattern: str, template: int = 24, max_search: int = 100) -> Optional[int]:
    """Find existing experiment by title pattern and template"""
    try:
        # Try to get recent experiments and check if any match our pattern
        # This is a workaround since elabapi might not support direct search
        # We'll check the most recent experiments first
        for offset in range(0, max_search, 50):  # Check in batches of 50
            try:
                # Try to get experiments with offset (this might not work with all elabapi versions)
                # For now, we'll return None and rely on the experiment_links approach
                break
            except Exception:
                continue
        return None
    except Exception:
        return None

def cmd_upload_test_run(run_label: Optional[str], title_prefix: str, category: Optional[int] = None, tags: Optional[str] = None, team: Optional[int] = None, experiment_links: Optional[List[int]] = None, item_links: Optional[List[int]] = None, update_existing: bool = True, update_experiment_id: Optional[int] = None) -> int:
    """Upload test run results (logs and detection_results) to elab"""
    label: str = run_label or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    
    # Default directories for test runs
    test_directories = ["logs", "detection_results"]
    
    # Check if directories exist
    existing_dirs = [d for d in test_directories if os.path.isdir(d)]
    if not existing_dirs:
        print("No test run directories found (logs, detection_results)", file=sys.stderr)
        return 4
    
    # Also check for test results summary file
    additional_files = []
    test_summary = "test_results_summary.yaml"
    if os.path.exists(test_summary):
        additional_files.append(test_summary)
        print(f"Found test results summary: {test_summary}")
    
    archives: List[str] = archive_directories(existing_dirs, label)
    if not archives:
        print("No directories found to archive", file=sys.stderr)
        return 4
    
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
    uploads_api: elabapi_python.UploadsApi = elabapi_python.UploadsApi(api_client)
    
    title: str = f"{title_prefix} {label}".strip()
    body: str = f"Test run results from test_single_particle.py: {', '.join(existing_dirs)}"
    if additional_files:
        body += f"\nAdditional files: {', '.join(additional_files)}"
    
    # Prepare experiment body with required fields
    experiment_body = {"title": title, "body": body, "template": 24}
    
    # Add optional metadata fields if provided (only add if they are valid)
    if category is not None and isinstance(category, int):
        experiment_body["category"] = category
    if tags is not None and isinstance(tags, str):
        experiment_body["tags_id"] = tags  # Use tags_id instead of tags
    if team is not None and isinstance(team, int):
        experiment_body["team"] = team
    
    # Add default metadata for LodeSTAR template if not provided
    if category is None:
        experiment_body["category"] = 5  # "Full Run" category
    if team is None:
        experiment_body["team"] = 1  # "Molecular Nanophotonics Group"
    if tags is None:
        experiment_body["tags_id"] = "155,156"  # Default LodeSTAR|ML tags
    
    # Check if we should try to update an existing experiment
    experiment_id: Optional[int] = None
    
    # First priority: if update_experiment_id is specified, use that
    if update_experiment_id is not None:
        try:
            existing_exp = experiments_api.get_experiment(update_experiment_id)
            if existing_exp and hasattr(existing_exp, 'template') and existing_exp.template == 24:
                experiment_id = update_experiment_id
                print(f"Will update specified experiment {experiment_id}")
                # Add this to experiment_links for proper linking
                if experiment_links is None:
                    experiment_links = []
                if update_experiment_id not in experiment_links:
                    experiment_links.append(update_experiment_id)
            else:
                print(f"Warning: Specified experiment {update_experiment_id} not found or has wrong template, will create new")
        except Exception as exc:
            print(f"Warning: Could not access specified experiment {update_experiment_id}: {exc}, will create new")
    
    # Second priority: if linking to experiments, try to update one of them
    if experiment_id is None and update_existing and experiment_links:
        for link_exp_id in experiment_links:
            try:
                existing_exp = experiments_api.get_experiment(link_exp_id)
                if existing_exp and hasattr(existing_exp, 'template') and existing_exp.template == 24:
                    experiment_id = link_exp_id
                    print(f"Found existing experiment {experiment_id} in links, will update instead of create new")
                    break
            except Exception:
                continue
    
    # If no experiment found through links, try to find by title pattern
    if experiment_id is None and update_existing:
        # Look for experiments with similar titles (e.g., "Test run results" pattern)
        base_title = title_prefix.strip()
        if base_title:
            found_exp_id = find_experiment_by_title_pattern(experiments_api, base_title, 24)
            if found_exp_id:
                experiment_id = found_exp_id
                print(f"Found existing experiment {experiment_id} with similar title, will update instead of create new")
                # Add this to experiment_links for proper linking
                if experiment_links is None:
                    experiment_links = []
                if found_exp_id not in experiment_links:
                    experiment_links.append(found_exp_id)
    
    if experiment_id is None:
        # Create new experiment
        try:
            exp_resp: Any = experiments_api.post_experiment_with_http_info(body=experiment_body)
        except Exception as exc:
            print(f"Failed to create experiment: {exc}", file=sys.stderr)
            return 5
        
        try:
            location_header: Optional[str] = exp_resp[2].get("Location")  # type: ignore[index]
            experiment_id: int = int(str(location_header).split("/").pop()) if location_header else int(exp_resp[0].id)  # type: ignore[attr-defined]
        except Exception:
            print_json({"unexpected_experiment_response": str(exp_resp)})
            return 6
    else:
        # Update existing experiment
        try:
            # Update the experiment with new body content
            update_body = {
                "title": title,
                "body": body
            }
            # Only update metadata if it's different
            if category is not None:
                update_body["category"] = category
            if team is not None:
                update_body["team"] = team
            if tags is not None:
                update_body["tags_id"] = tags
            
            experiments_api.put_experiment(experiment_id, body=update_body)
            print(f"Updated existing experiment {experiment_id}")
        except Exception as exc:
            print(f"Warning: Failed to update experiment {experiment_id}: {exc}")
            # Continue with uploads even if update fails
    
    try:
        location_header: Optional[str] = exp_resp[2].get("Location")  # type: ignore[index]
        experiment_id: int = int(str(location_header).split("/").pop()) if location_header else int(exp_resp[0].id)  # type: ignore[attr-defined]
    except Exception:
        print_json({"unexpected_experiment_response": str(exp_resp)})
        return 6
    
    # Get the actual tags from the raw response since Python client has a bug
    actual_tags = None
    if hasattr(exp_resp[0], 'tags') and exp_resp[0].tags:
        actual_tags = exp_resp[0].tags
    elif hasattr(exp_resp[0], 'tags_id') and exp_resp[0].tags_id:
        actual_tags = exp_resp[0].tags_id
    
    # Upload archived directories
    for archive in archives:
        try:
            uploads_api.post_upload(
                id=experiment_id, 
                file=archive, 
                entity_type="experiments"
            )
        except Exception as exc:
            print(f"Failed to upload {archive} to experiment {experiment_id}: {exc}", file=sys.stderr)
            return 7
    
    # Upload additional files
    for file_path in additional_files:
        try:
            uploads_api.post_upload(
                id=experiment_id, 
                file=file_path, 
                entity_type="experiments"
            )
        except Exception as exc:
            print(f"Failed to upload {file_path} to experiment {experiment_id}: {exc}", file=sys.stderr)
            return 7
    
    # Link resources if specified
    link_results = None
    if experiment_links or item_links:
        try:
            link_results = link_resources_to_experiment(experiment_id, experiment_links, item_links)
        except Exception as exc:
            print(f"Warning: Failed to link resources: {exc}", file=sys.stderr)
            link_results = {"errors": [str(exc)]}
    
    print_json({
        "experiment_id": experiment_id, 
        "uploaded": archives + additional_files,
        "test_run_type": "single_particle_detection",
        "directories_archived": existing_dirs,
        "additional_files": additional_files,
        "experiment_metadata": {
            "title": title,
            "category": experiment_body.get("category"),
            "tags": actual_tags,
            "team": experiment_body.get("team")
        }
    })
    
    if link_results:
        print_json({
            "linked_resources": link_results
        })
    
    return 0


def link_resources_to_experiment(experiment_id: int, experiment_links: Optional[List[int]] = None, item_links: Optional[List[int]] = None) -> Dict[str, Any]:
    """Link resources (experiments and items) to an experiment"""
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_links_api: elabapi_python.LinksToExperimentsApi = elabapi_python.LinksToExperimentsApi(api_client)
    items_links_api: elabapi_python.LinksToItemsApi = elabapi_python.LinksToItemsApi(api_client)
    
    results: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "experiment_links": [],
        "item_links": [],
        "errors": []
    }
    
    # Link experiments
    if experiment_links:
        for link_exp_id in experiment_links:
            try:
                link_body = elabapi_python.ExperimentsLinksSubidBody(action="duplicate")
                experiments_links_api.post_entity_experiments_links(
                    entity_type="experiments",
                    id=experiment_id,
                    subid=link_exp_id,
                    body=link_body
                )
                results["experiment_links"].append({
                    "id": link_exp_id,
                    "status": "linked",
                    "type": "experiment"
                })
            except Exception as exc:
                error_msg = f"Failed to link experiment {link_exp_id}: {exc}"
                results["errors"].append(error_msg)
                print(error_msg, file=sys.stderr)
    
    # Link items
    if item_links:
        for link_item_id in item_links:
            try:
                link_body = elabapi_python.ItemsLinksSubidBody(action="duplicate")
                items_links_api.post_entity_items_links(
                    entity_type="experiments",
                    id=experiment_id,
                    subid=link_item_id,
                    body=link_body
                )
                results["item_links"].append({
                    "id": link_item_id,
                    "status": "linked",
                    "type": "item"
                })
            except Exception as exc:
                error_msg = f"Failed to link item {link_item_id}: {exc}"
                results["errors"].append(error_msg)
                print(error_msg, file=sys.stderr)
    
    return results


def cmd_link_resources(experiment_id: int, experiment_links: Optional[List[int]] = None, item_links: Optional[List[int]] = None) -> int:
    """Link resources to an existing experiment"""
    if not experiment_links and not item_links:
        print("No resources specified to link", file=sys.stderr)
        return 1
    
    try:
        results = link_resources_to_experiment(experiment_id, experiment_links, item_links)
        print_json(results)
        
        if results["errors"]:
            return 2  # Partial success with errors
        return 0
        
    except Exception as exc:
        print(f"Failed to link resources: {exc}", file=sys.stderr)
        return 3


def cmd_create_experiment_with_links(title: str, body: str, template: int, category: Optional[int] = None, 
                                   tags: Optional[str] = None, team: Optional[int] = None,
                                   experiment_links: Optional[List[int]] = None, 
                                   item_links: Optional[List[int]] = None) -> int:
    """Create an experiment and link resources to it"""
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
    
    # Prepare experiment body
    experiment_body = {"title": title, "body": body, "template": template}
    
    # Add optional metadata fields
    if category is not None and isinstance(category, int):
        experiment_body["category"] = category
    if tags is not None and isinstance(tags, str):
        experiment_body["tags_id"] = tags
    if team is not None and isinstance(team, int):
        experiment_body["team"] = team
    
    # Add default metadata for LodeSTAR template if not provided
    if template == 24:
        if category is None:
            experiment_body["category"] = 5  # "Full Run" category
        if team is None:
            experiment_body["team"] = 1  # "Molecular Nanophotonics Group"
        if tags is None:
            experiment_body["tags_id"] = "155,156"  # Default LodeSTAR|ML tags
    
    try:
        exp_resp: Any = experiments_api.post_experiment_with_http_info(body=experiment_body)
    except Exception as exc:
        print(f"Failed to create experiment: {exc}", file=sys.stderr)
        return 1
    
    try:
        location_header: Optional[str] = exp_resp[2].get("Location")
        experiment_id: int = int(str(location_header).split("/").pop()) if location_header else int(exp_resp[0].id)
    except Exception:
        print_json({"unexpected_experiment_response": str(exp_resp)})
        return 2
    
    # Link resources if specified
    link_results = None
    if experiment_links or item_links:
        try:
            link_results = link_resources_to_experiment(experiment_id, experiment_links, item_links)
        except Exception as exc:
            print(f"Warning: Failed to link resources: {exc}", file=sys.stderr)
            link_results = {"errors": [str(exc)]}
    
    # Prepare output
    output = {
        "experiment_id": experiment_id,
        "title": title,
        "template": template,
        "metadata": experiment_body
    }
    
    if link_results:
        output["linked_resources"] = link_results
    
    print_json(output)
    return 0


def cmd_clone_experiment(source_id: int, template: int, title_override: Optional[str] = None) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
    items_links_api: elabapi_python.LinksToItemsApi = elabapi_python.LinksToItemsApi(api_client)

    # Fetch source experiment raw
    configuration = api_client.configuration  # type: ignore[attr-defined]
    raw_exp, raw_err = _fetch_experiment_raw(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=source_id,
    )
    if raw_exp is None:
        print(f"Failed to fetch source experiment {source_id}: {raw_err}", file=sys.stderr)
        return 2

    title: str = title_override or str(raw_exp.get("title", f"Clone of {source_id}"))
    body: str = str(raw_exp.get("body", ""))
    category = raw_exp.get("category")
    team = raw_exp.get("team")
    tags_id = raw_exp.get("tags_id")
    item_ids = []
    try:
        for link in raw_exp.get("items_links", []) or []:
            if isinstance(link, dict) and "entityid" in link:
                item_ids.append(int(link["entityid"]))
    except Exception:
        item_ids = []

    # Create the new experiment
    experiment_body: Dict[str, Any] = {"title": title, "body": body, "template": template}
    if isinstance(category, int):
        experiment_body["category"] = category
    if isinstance(team, int):
        experiment_body["team"] = team
    if isinstance(tags_id, str):
        experiment_body["tags_id"] = tags_id

    try:
        exp_resp: Any = experiments_api.post_experiment_with_http_info(body=experiment_body)
    except Exception as exc:
        print(f"Failed to create experiment: {exc}", file=sys.stderr)
        return 3

    try:
        location_header: Optional[str] = exp_resp[2].get("Location")  # type: ignore[index]
        new_experiment_id: int = int(str(location_header).split("/").pop()) if location_header else int(exp_resp[0].id)  # type: ignore[attr-defined]
    except Exception:
        print_json({"unexpected_experiment_response": str(exp_resp)})
        return 4

    # Link items
    for link_item_id in item_ids:
        try:
            link_body = elabapi_python.ItemsLinksSubidBody(action="duplicate")
            items_links_api.post_entity_items_links(
                entity_type="experiments",
                id=new_experiment_id,
                subid=link_item_id,
                body=link_body,
            )
        except Exception as exc:
            print(f"Warning: Failed to link item {link_item_id}: {exc}", file=sys.stderr)

    print_json({
        "cloned_from": source_id,
        "new_experiment_id": new_experiment_id,
        "metadata": experiment_body,
        "linked_items": item_ids,
    })
    return 0


def _get_experiment_raw_for_compare(api_client: elabapi_python.ApiClient, exp_id: int) -> Dict[str, Any]:
    configuration = api_client.configuration  # type: ignore[attr-defined]
    raw_exp, raw_err = _fetch_experiment_raw(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=exp_id,
    )
    if raw_exp is None:
        return {"error": raw_err or "unknown"}
    return raw_exp


def cmd_compare_experiments(a_id: int, b_id: int) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    a = _get_experiment_raw_for_compare(api_client, a_id)
    b = _get_experiment_raw_for_compare(api_client, b_id)

    def summarize(exp: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in exp:
            return exp
        return {
            "title": exp.get("title"),
            "category": exp.get("category"),
            "team": exp.get("team"),
            "tags_id": exp.get("tags_id"),
            "items_links_entityids": [link.get("entityid") for link in (exp.get("items_links") or []) if isinstance(link, dict)],
            "has_uploads": bool(exp.get("uploads")),
            "body_len": len(str(exp.get("body", ""))),
        }

    out = {
        "a_id": a_id,
        "b_id": b_id,
        "a_summary": summarize(a),
        "b_summary": summarize(b),
        "equal": summarize(a) == summarize(b),
    }
    print_json(out)
    return 0


def _build_web_base_from_api_host(host_url: str) -> str:
    base = host_url.rstrip('/')
    if base.endswith('/api/v2'):
        return base[:-7]
    if '/api/' in base:
        return base[: base.find('/api/')]
    return base


def cmd_clone_experiment_full(source_id: int, template: int = 24, title_override: Optional[str] = None) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
    items_links_api: elabapi_python.LinksToItemsApi = elabapi_python.LinksToItemsApi(api_client)
    uploads_api: elabapi_python.UploadsApi = elabapi_python.UploadsApi(api_client)

    configuration = api_client.configuration  # type: ignore[attr-defined]
    raw_exp, raw_err = _fetch_experiment_raw(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=source_id,
    )
    if raw_exp is None:
        print(f"Failed to fetch source experiment {source_id}: {raw_err}", file=sys.stderr)
        return 2

    title: str = title_override or str(raw_exp.get("title", f"Clone of {source_id}"))
    body: str = str(raw_exp.get("body", ""))
    category = raw_exp.get("category")
    team = raw_exp.get("team")
    tags_id = raw_exp.get("tags_id")

    # Create the new experiment with exact metadata
    experiment_body: Dict[str, Any] = {"title": title, "body": body, "template": template}
    if isinstance(category, int):
        experiment_body["category"] = category
    if isinstance(team, int):
        experiment_body["team"] = team
    if isinstance(tags_id, str):
        experiment_body["tags_id"] = tags_id

    try:
        exp_resp: Any = experiments_api.post_experiment_with_http_info(body=experiment_body)
    except Exception as exc:
        print(f"Failed to create experiment: {exc}", file=sys.stderr)
        return 3

    try:
        location_header: Optional[str] = exp_resp[2].get("Location")  # type: ignore[index]
        new_experiment_id: int = int(str(location_header).split("/").pop()) if location_header else int(exp_resp[0].id)  # type: ignore[attr-defined]
    except Exception:
        print_json({"unexpected_experiment_response": str(exp_resp)})
        return 4

    # Link items exactly
    item_ids: List[int] = []
    for link in (raw_exp.get("items_links") or []):
        if isinstance(link, dict) and "entityid" in link:
            item_ids.append(int(link["entityid"]))
    for link_item_id in item_ids:
        try:
            link_body = elabapi_python.ItemsLinksSubidBody(action="duplicate")
            items_links_api.post_entity_items_links(
                entity_type="experiments",
                id=new_experiment_id,
                subid=link_item_id,
                body=link_body,
            )
        except Exception as exc:
            print(f"Warning: Failed to link item {link_item_id}: {exc}", file=sys.stderr)

    # Copy uploads by downloading and re-uploading
    web_base = _build_web_base_from_api_host(str(configuration.host))
    headers = {"Authorization": os.environ.get("ELAB_API_KEY", "")}
    copied_uploads: List[str] = []
    for up in (raw_exp.get("uploads") or []):
        try:
            long_name = up.get("long_name")
            real_name = up.get("real_name")
            storage = up.get("storage", 1)
            if not long_name or not real_name:
                continue
            download_url = f"{web_base}/app/download.php?f={urllib.parse.quote(long_name)}&name={urllib.parse.quote(real_name)}&storage={storage}"
            resp = requests.get(download_url, headers=headers, verify=bool(configuration.verify_ssl))
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False) as tmpf:
                tmpf.write(resp.content)
                tmp_path = tmpf.name
            uploads_api.post_upload(id=new_experiment_id, file=tmp_path, entity_type="experiments")
            copied_uploads.append(real_name)
        except Exception as exc:
            print(f"Warning: Failed to copy upload {up}: {exc}", file=sys.stderr)
            continue

    print_json({
        "cloned_from": source_id,
        "new_experiment_id": new_experiment_id,
        "metadata": experiment_body,
        "linked_items": item_ids,
        "copied_uploads": copied_uploads,
    })
    return 0


def _patch_experiment(host_url: str, api_key: str, verify_ssl: bool, experiment_id: int, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    try:
        url = host_url.rstrip('/') + f"/experiments/{experiment_id}"
        headers = {"Authorization": api_key, "Content-Type": "application/json", "Accept": "application/json"}
        resp = requests.patch(url, headers=headers, json=payload, verify=verify_ssl)
        if resp.status_code not in (200, 204):
            return False, f"HTTP {resp.status_code}: {resp.text}"
        return True, None
    except Exception as exc:
        return False, str(exc)


def cmd_sync_experiment_from(source_id: int, target_id: int) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    configuration = api_client.configuration  # type: ignore[attr-defined]
    src, err = _fetch_experiment_raw(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=source_id,
    )
    if src is None:
        print(f"Failed to fetch source {source_id}: {err}", file=sys.stderr)
        return 2
    # Prepare payload: title, body, tags_id
    payload: Dict[str, Any] = {}
    if isinstance(src.get("title"), str):
        payload["title"] = src.get("title")
    # Prefer 'body' raw
    if src.get("body") is not None:
        payload["body"] = src.get("body")
    elif src.get("body_html") is not None:
        payload["body_html"] = src.get("body_html")
    if isinstance(src.get("tags_id"), str):
        payload["tags_id"] = src.get("tags_id")

    ok, perr = _patch_experiment(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=target_id,
        payload=payload,
    )
    if not ok:
        print(f"Failed to patch target {target_id}: {perr}", file=sys.stderr)
        return 3
    print_json({"synced_from": source_id, "target": target_id, "applied": payload})
    return 0


def cmd_reupload_files_from(source_id: int, target_id: int) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    uploads_api: elabapi_python.UploadsApi = elabapi_python.UploadsApi(api_client)
    configuration = api_client.configuration  # type: ignore[attr-defined]
    src, err = _fetch_experiment_raw(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=source_id,
    )
    if src is None:
        print(f"Failed to fetch source {source_id}: {err}", file=sys.stderr)
        return 2
    web_base = _build_web_base_from_api_host(str(configuration.host))
    headers = {"Authorization": os.environ.get("ELAB_API_KEY", "")}
    copied: List[str] = []
    for up in (src.get("uploads") or []):
        long_name = up.get("long_name")
        real_name = up.get("real_name")
        storage = up.get("storage", 1)
        if not long_name or not real_name:
            continue
        try:
            url = f"{web_base}/app/download.php?f={urllib.parse.quote(long_name)}&name={urllib.parse.quote(real_name)}&storage={storage}"
            r = requests.get(url, headers=headers, verify=bool(configuration.verify_ssl))
            r.raise_for_status()
            with tempfile.TemporaryDirectory() as td:
                out_path = os.path.join(td, real_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'wb') as f:
                    f.write(r.content)
                # Upload; this API uses filename from the basename of path
                uploads_api.post_upload(id=target_id, file=out_path, entity_type="experiments")
                copied.append(real_name)
        except Exception as exc:
            print(f"Warning: failed to reupload {real_name}: {exc}", file=sys.stderr)
    print_json({"reuploaded_from": source_id, "target": target_id, "copied": copied})
    return 0


def cmd_set_experiment_body(experiment_id: int, body_file: str) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    configuration = api_client.configuration  # type: ignore[attr-defined]
    try:
        with open(body_file, "r", encoding="utf-8") as fh:
            body_html = fh.read()
    except Exception as exc:
        print(f"Failed to read body file {body_file}: {exc}", file=sys.stderr)
        return 2
    ok, err = _patch_experiment(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=experiment_id,
        payload={"body": body_html},
    )
    if not ok:
        print(f"Failed to set body for experiment {experiment_id}: {err}", file=sys.stderr)
        return 3
    print_json({"experiment_id": experiment_id, "body_file": body_file, "status": "updated"})
    return 0


def _post_experiment_by_id_full(api_client: elabapi_python.ApiClient, experiment_id: int, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    try:
        experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
        experiments_api.post_experiment_by_id(experiment_id, body=payload)
        return True, None
    except Exception as exc:
        return False, str(exc)


def cmd_set_experiment_tags(experiment_id: int, tags_id: str) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
    configuration = api_client.configuration  # type: ignore[attr-defined]
    # Fetch existing experiment to preserve required fields
    raw_exp, raw_err = _fetch_experiment_raw(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=experiment_id,
    )
    if raw_exp is None:
        print(f"Failed to fetch experiment {experiment_id}: {raw_err}", file=sys.stderr)
        return 2
    # Resolve tag names alongside IDs to maximize server acceptance
    tag_names: Optional[str] = None
    try:
        tags_api: elabapi_python.TagsApi = elabapi_python.TagsApi(api_client)
        resolved_names: List[str] = []
        for tid_s in str(tags_id).split(','):
            tid_s = tid_s.strip()
            if not tid_s:
                continue
            try:
                t = tags_api.read_tag(int(tid_s))
                name = getattr(t, 'tag', None) or getattr(t, 'name', None)
                if isinstance(name, str) and name:
                    resolved_names.append(name)
            except Exception:
                continue
        if resolved_names:
            tag_names = "|".join(resolved_names)
    except Exception:
        tag_names = None
    payload: Dict[str, Any] = {
        "title": raw_exp.get("title"),
        "template": raw_exp.get("template", 24),
        "category": raw_exp.get("category"),
        "team": raw_exp.get("team"),
        "body": raw_exp.get("body") or raw_exp.get("body_html") or "",
        "tags_id": tags_id,
    }
    # Ensure valid template
    try:
        if not isinstance(payload["template"], int) or payload["template"] < 1:
            payload["template"] = 24
    except Exception:
        payload["template"] = 24
    if isinstance(tag_names, str) and tag_names:
        payload["tags"] = tag_names
    ok, perr = _post_experiment_by_id_full(api_client, experiment_id, payload)
    if ok:
        # Verify on server
        new_raw, nerr = _fetch_experiment_raw(
            host_url=str(configuration.host),
            api_key=os.environ.get("ELAB_API_KEY", ""),
            verify_ssl=bool(configuration.verify_ssl),
            experiment_id=experiment_id,
        )
        if isinstance(new_raw, dict) and (new_raw.get("tags_id") or new_raw.get("tags")):
            print_json({
                "experiment_id": experiment_id,
                "tags_id": new_raw.get("tags_id"),
                "tags": new_raw.get("tags"),
                "status": "updated_via_post_by_id"
            })
            return 0
        print(f"Server did not reflect tags after update; raw_err={nerr}", file=sys.stderr)
    # Fallback: try direct PATCH
    ok2, perr2 = _patch_experiment(
        host_url=str(configuration.host),
        api_key=os.environ.get("ELAB_API_KEY", ""),
        verify_ssl=bool(configuration.verify_ssl),
        experiment_id=experiment_id,
        payload={"tags_id": tags_id},
    )
    if ok2:
        new_raw, _ = _fetch_experiment_raw(
            host_url=str(configuration.host),
            api_key=os.environ.get("ELAB_API_KEY", ""),
            verify_ssl=bool(configuration.verify_ssl),
            experiment_id=experiment_id,
        )
        print_json({
            "experiment_id": experiment_id,
            "tags_id": isinstance(new_raw, dict) and new_raw.get("tags_id") or tags_id,
            "tags": isinstance(new_raw, dict) and new_raw.get("tags") or None,
            "status": "updated_via_patch"
        })
        return 0
    print(f"Failed to set tags for experiment {experiment_id}: {perr}; patch_fallback={perr2}", file=sys.stderr)
    return 3


def cmd_attach_uploads(experiment_id: int, files: List[str]) -> int:
    api_client: elabapi_python.ApiClient = build_api_client()
    uploads_api: elabapi_python.UploadsApi = elabapi_python.UploadsApi(api_client)
    uploaded: List[str] = []
    for path in files:
        try:
            if not os.path.isfile(path):
                print(f"Warning: file not found: {path}", file=sys.stderr)
                continue
            uploads_api.post_upload(id=experiment_id, file=path, entity_type="experiments")
            uploaded.append(os.path.basename(path))
        except Exception as exc:
            print(f"Warning: failed to upload {path}: {exc}", file=sys.stderr)
    print_json({"experiment_id": experiment_id, "uploaded": uploaded})
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="elab-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_check = subparsers.add_parser("resource-check")
    p_check.add_argument("--id", dest="resource_id", type=int, required=True)
    p_check.add_argument("--expect", dest="expected_metadata_keys", nargs="*", default=None)

    p_exp_check = subparsers.add_parser("experiment-check")
    p_exp_check.add_argument("--id", dest="experiment_id", type=int, required=True)

    p_upload = subparsers.add_parser("upload-run")
    p_upload.add_argument("--label", dest="run_label", type=str, required=False, default=None)
    p_upload.add_argument("--dirs", dest="directories", nargs="+", default=["logs", "data", "detection_results"])
    p_upload.add_argument("--title-prefix", dest="title_prefix", type=str, required=False, default="Run artifacts")

    p_upload_test = subparsers.add_parser("upload-test-run", 
                                         help="Upload test run results from test_single_particle.py to elab")
    p_upload_test.add_argument("--label", dest="run_label", type=str, required=False, default=None,
                              help="Custom label for the test run (default: timestamp)")
    p_upload_test.add_argument("--title-prefix", dest="title_prefix", type=str, required=False, 
                              default="Test run results", help="Prefix for the experiment title")
    p_upload_test.add_argument("--category", dest="category", type=int, required=False, default=None,
                              help="Experiment category ID (e.g., 5 for 'Full Run')")
    p_upload_test.add_argument("--tags", dest="tags", type=str, required=False, default=None,
                              help="Experiment tag IDs (e.g., '155,156' for 'LodeSTAR|ML')")
    p_upload_test.add_argument("--team", dest="team", type=int, required=False, default=None,
                              help="Team ID (e.g., 1 for 'Molecular Nanophotonics Group')")
    p_upload_test.add_argument("--experiments", dest="experiment_links", type=int, nargs="*", default=None,
                              help="List of experiment IDs to link to this experiment")
    p_upload_test.add_argument("--items", dest="item_links", type=int, nargs="*", default=None,
                              help="List of item IDs to link to this experiment")
    p_upload_test.add_argument("--no-update", dest="update_existing", action="store_false", default=True,
                              help="Don't try to update existing experiments, always create new ones")
    p_upload_test.add_argument("--update-experiment", dest="update_experiment_id", type=int, default=None,
                              help="Update a specific existing experiment ID instead of creating new one")

    p_link = subparsers.add_parser("link-resources", 
                                  help="Link resources (experiments/items) to an existing experiment")
    p_link.add_argument("--experiment-id", dest="experiment_id", type=int, required=True,
                       help="ID of the experiment to link resources to")
    p_link.add_argument("--experiments", dest="experiment_links", type=int, nargs="*", default=None,
                       help="List of experiment IDs to link")
    p_link.add_argument("--items", dest="item_links", type=int, nargs="*", default=None,
                       help="List of item IDs to link")

    p_create_with_links = subparsers.add_parser("create-with-links",
                                               help="Create an experiment and link resources to it")
    p_create_with_links.add_argument("--title", dest="title", type=str, required=True,
                                   help="Experiment title")
    p_create_with_links.add_argument("--body", dest="body", type=str, required=True,
                                   help="Experiment body/description")
    p_create_with_links.add_argument("--template", dest="template", type=int, required=True,
                                   help="Template ID for the experiment")
    p_create_with_links.add_argument("--category", dest="category", type=int, required=False, default=None,
                                   help="Experiment category ID")
    p_create_with_links.add_argument("--tags", dest="tags", type=str, required=False, default=None,
                                   help="Experiment tag IDs (e.g., '155,156')")
    p_create_with_links.add_argument("--team", dest="team", type=int, required=False, default=None,
                                   help="Team ID")
    p_create_with_links.add_argument("--experiments", dest="experiment_links", type=int, nargs="*", default=None,
                                   help="List of experiment IDs to link")
    p_create_with_links.add_argument("--items", dest="item_links", type=int, nargs="*", default=None,
                                   help="List of item IDs to link")

    p_clone = subparsers.add_parser("clone-experiment",
                                         help="Clone an existing experiment to a new one with a different template")
    p_clone.add_argument("--source", dest="source_id", type=int, required=True,
                              help="ID of the experiment to clone from")
    p_clone.add_argument("--template", dest="template", type=int, required=True,
                              help="Template ID for the new experiment")
    p_clone.add_argument("--title", dest="title_override", type=str, required=False,
                              help="Optional override for the new experiment's title")

    p_clone_full = subparsers.add_parser("clone-experiment-full",
                                         help="Clone an existing experiment to a new one, copying title, body, category, team, tags_id, items_links, and uploads by downloading and re-uploading.")
    p_clone_full.add_argument("--source", dest="source_id", type=int, required=True,
                              help="ID of the experiment to clone from")
    p_clone_full.add_argument("--template", dest="template", type=int, required=False, default=24)
    p_clone_full.add_argument("--title", dest="title_override", type=str, required=False, default=None)

    p_compare = subparsers.add_parser("compare-experiments",
                                          help="Compare two existing experiments for metadata and uploads")
    p_compare.add_argument("--a", dest="a_id", type=int, required=True,
                                help="ID of the first experiment to compare")
    p_compare.add_argument("--b", dest="b_id", type=int, required=True,
                                help="ID of the second experiment to compare")

    p_sync = subparsers.add_parser("sync-experiment-from")
    p_sync.add_argument("--source", dest="source_id", type=int, required=True)
    p_sync.add_argument("--target", dest="target_id", type=int, required=True)

    p_reup = subparsers.add_parser("reupload-files-from")
    p_reup.add_argument("--source", dest="source_id", type=int, required=True)
    p_reup.add_argument("--target", dest="target_id", type=int, required=True)

    p_set_body = subparsers.add_parser("set-experiment-body")
    p_set_body.add_argument("--id", dest="experiment_id", type=int, required=True)
    p_set_body.add_argument("--file", dest="body_file", type=str, required=True)

    p_set_tags = subparsers.add_parser("set-experiment-tags")
    p_set_tags.add_argument("--id", dest="experiment_id", type=int, required=True)
    p_set_tags.add_argument("--tags", dest="tags_id", type=str, required=True)

    p_attach = subparsers.add_parser("attach-uploads")
    p_attach.add_argument("--id", dest="experiment_id", type=int, required=True)
    p_attach.add_argument("--files", dest="files", nargs="+", required=True)

    args = parser.parse_args(argv)

    if args.command == "resource-check":
        return cmd_resource_check(resource_id=args.resource_id, expected_metadata_keys=args.expected_metadata_keys)
    if args.command == "experiment-check":
        return cmd_experiment_check(experiment_id=args.experiment_id)
    if args.command == "upload-run":
        return cmd_upload_run(run_label=args.run_label, directories=args.directories, title_prefix=args.title_prefix)
    if args.command == "upload-test-run":
        return cmd_upload_test_run(run_label=args.run_label, title_prefix=args.title_prefix, category=args.category, tags=args.tags, team=args.team, experiment_links=args.experiment_links, item_links=args.item_links, update_existing=args.update_existing, update_experiment_id=args.update_experiment_id)
    if args.command == "link-resources":
        return cmd_link_resources(experiment_id=args.experiment_id, experiment_links=args.experiment_links, item_links=args.item_links)
    if args.command == "create-with-links":
        return cmd_create_experiment_with_links(title=args.title, body=args.body, template=args.template, category=args.category, tags=args.tags, team=args.team, experiment_links=args.experiment_links, item_links=args.item_links)
    if args.command == "clone-experiment":
        return cmd_clone_experiment(source_id=args.source_id, template=args.template, title_override=args.title_override)
    if args.command == "clone-experiment-full":
        return cmd_clone_experiment_full(source_id=args.source_id, template=args.template, title_override=args.title_override)
    if args.command == "compare-experiments":
        return cmd_compare_experiments(a_id=args.a_id, b_id=args.b_id)
    if args.command == "sync-experiment-from":
        return cmd_sync_experiment_from(source_id=args.source_id, target_id=args.target_id)
    if args.command == "reupload-files-from":
        return cmd_reupload_files_from(source_id=args.source_id, target_id=args.target_id)
    if args.command == "set-experiment-body":
        return cmd_set_experiment_body(experiment_id=args.experiment_id, body_file=args.body_file)
    if args.command == "set-experiment-tags":
        return cmd_set_experiment_tags(experiment_id=args.experiment_id, tags_id=args.tags_id)
    if args.command == "attach-uploads":
        return cmd_attach_uploads(experiment_id=args.experiment_id, files=args.files)
    return 1


if __name__ == "__main__":
    sys.exit(main())
