import argparse
import json
import os
import sys
import tarfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import elabapi_python
import urllib3


def build_api_client() -> elabapi_python.ApiClient:
    host_url: Optional[str] = os.environ.get("ELAB_HOST_URL")
    api_key: Optional[str] = os.environ.get("ELAB_API_KEY")
    verify_ssl_env: str = os.environ.get("ELAB_VERIFY_SSL", "true").strip().lower()
    
    if not host_url or not api_key:
        raise RuntimeError("Missing ELAB_HOST_URL or ELAB_API_KEY in environment")
    
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


def cmd_upload_training_results(run_label: Optional[str], title_prefix: str, 
                               category: Optional[int] = None, team: Optional[int] = None,
                               experiment_links: Optional[List[int]] = None, 
                               item_links: Optional[List[int]] = None) -> int:
    """Upload training results to elab"""
    label: str = run_label or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    
    training_directories = ["logs", "checkpoints", "models"]
    existing_dirs = [d for d in training_directories if os.path.isdir(d)]
    
    if not existing_dirs:
        print("No training directories found (logs, checkpoints, models)", file=sys.stderr)
        return 1
    
    archives: List[str] = archive_directories(existing_dirs, label)
    if not archives:
        print("No directories found to archive", file=sys.stderr)
        return 1
    
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
    uploads_api: elabapi_python.UploadsApi = elabapi_python.UploadsApi(api_client)
    
    title: str = f"{title_prefix} {label}".strip()
    body: str = f"Training results: {', '.join(existing_dirs)}"
    
    experiment_body = {
        "title": title, 
        "body": body, 
        "template": 24,
        "category": category or 5,
        "team": team or 1
    }
    
    try:
        exp_resp: Any = experiments_api.post_experiment_with_http_info(body=experiment_body)
        location_header: Optional[str] = exp_resp[2].get("Location")
        experiment_id: int = int(str(location_header).split("/").pop()) if location_header else int(exp_resp[0].id)
    except Exception as exc:
        print(f"Failed to create experiment: {exc}", file=sys.stderr)
        return 2
    
    for archive in archives:
        try:
            uploads_api.post_upload(id=experiment_id, file=archive, entity_type="experiments")
        except Exception as exc:
            print(f"Failed to upload {archive}: {exc}", file=sys.stderr)
            return 3
    
    print_json({
        "experiment_id": experiment_id,
        "title": title,
        "uploaded": archives,
        "training_directories": existing_dirs
    })
    
    return 0


def cmd_upload_test_results(run_label: Optional[str], title_prefix: str,
                           category: Optional[int] = None, team: Optional[int] = None,
                           experiment_links: Optional[List[int]] = None,
                           item_links: Optional[List[int]] = None) -> int:
    """Upload test results to elab"""
    label: str = run_label or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    
    test_directories = ["logs", "detection_results"]
    existing_dirs = [d for d in test_directories if os.path.isdir(d)]
    
    if not existing_dirs:
        print("No test directories found (logs, detection_results)", file=sys.stderr)
        return 1
    
    additional_files = []
    test_summary = "test_results_summary.yaml"
    if os.path.exists(test_summary):
        additional_files.append(test_summary)
    
    archives: List[str] = archive_directories(existing_dirs, label)
    if not archives:
        print("No directories found to archive", file=sys.stderr)
        return 1
    
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_api: elabapi_python.ExperimentsApi = elabapi_python.ExperimentsApi(api_client)
    uploads_api: elabapi_python.UploadsApi = elabapi_python.UploadsApi(api_client)
    
    title: str = f"{title_prefix} {label}".strip()
    body: str = f"Test results: {', '.join(existing_dirs)}"
    if additional_files:
        body += f"\nAdditional files: {', '.join(additional_files)}"
    
    experiment_body = {
        "title": title,
        "body": body,
        "template": 24,
        "category": category or 5,
        "team": team or 1
    }
    
    try:
        exp_resp: Any = experiments_api.post_experiment_with_http_info(body=experiment_body)
        location_header: Optional[str] = exp_resp[2].get("Location")
        experiment_id: int = int(str(location_header).split("/").pop()) if location_header else int(exp_resp[0].id)
    except Exception as exc:
        print(f"Failed to create experiment: {exc}", file=sys.stderr)
        return 2
    
    for archive in archives:
        try:
            uploads_api.post_upload(id=experiment_id, file=archive, entity_type="experiments")
        except Exception as exc:
            print(f"Failed to upload {archive}: {exc}", file=sys.stderr)
            return 3
    
    for file_path in additional_files:
        try:
            uploads_api.post_upload(id=experiment_id, file=file_path, entity_type="experiments")
        except Exception as exc:
            print(f"Failed to upload {file_path}: {exc}", file=sys.stderr)
            return 3
    
    print_json({
        "experiment_id": experiment_id,
        "title": title,
        "uploaded": archives + additional_files,
        "test_directories": existing_dirs,
        "additional_files": additional_files
    })
    
    return 0


def cmd_link_resources(experiment_id: int, experiment_links: Optional[List[int]] = None, 
                      item_links: Optional[List[int]] = None) -> int:
    """Link resources to an existing experiment"""
    if not experiment_links and not item_links:
        print("No resources specified to link", file=sys.stderr)
        return 1
    
    api_client: elabapi_python.ApiClient = build_api_client()
    experiments_links_api: elabapi_python.LinksToExperimentsApi = elabapi_python.LinksToExperimentsApi(api_client)
    items_links_api: elabapi_python.LinksToItemsApi = elabapi_python.LinksToItemsApi(api_client)
    
    results: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "experiment_links": [],
        "item_links": [],
        "errors": []
    }
    
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
                results["experiment_links"].append({"id": link_exp_id, "status": "linked"})
            except Exception as exc:
                error_msg = f"Failed to link experiment {link_exp_id}: {exc}"
                results["errors"].append(error_msg)
    
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
                results["item_links"].append({"id": link_item_id, "status": "linked"})
            except Exception as exc:
                error_msg = f"Failed to link item {link_item_id}: {exc}"
                results["errors"].append(error_msg)
    
    print_json(results)
    return 0 if not results["errors"] else 2


def main(argv: Optional[List[str]] = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="elab-cli-simple")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_training = subparsers.add_parser("upload-training", 
                                      help="Upload training results to elab")
    p_training.add_argument("--label", dest="run_label", type=str, required=False, default=None,
                           help="Custom label for the training run (default: timestamp)")
    p_training.add_argument("--title-prefix", dest="title_prefix", type=str, required=False, 
                           default="Training Results", help="Prefix for the experiment title")
    p_training.add_argument("--category", dest="category", type=int, required=False, default=None,
                           help="Experiment category ID (e.g., 5 for 'Full Run')")
    p_training.add_argument("--team", dest="team", type=int, required=False, default=None,
                           help="Team ID (e.g., 1 for 'Molecular Nanophotonics Group')")
    p_training.add_argument("--experiments", dest="experiment_links", type=int, nargs="*", default=None,
                           help="List of experiment IDs to link")
    p_training.add_argument("--items", dest="item_links", type=int, nargs="*", default=None,
                           help="List of item IDs to link")

    p_test = subparsers.add_parser("upload-test", 
                                  help="Upload test results to elab")
    p_test.add_argument("--label", dest="run_label", type=str, required=False, default=None,
                       help="Custom label for the test run (default: timestamp)")
    p_test.add_argument("--title-prefix", dest="title_prefix", type=str, required=False, 
                       default="Test Results", help="Prefix for the experiment title")
    p_test.add_argument("--category", dest="category", type=int, required=False, default=None,
                       help="Experiment category ID (e.g., 5 for 'Full Run')")
    p_test.add_argument("--team", dest="team", type=int, required=False, default=None,
                       help="Team ID (e.g., 1 for 'Molecular Nanophotonics Group')")
    p_test.add_argument("--experiments", dest="experiment_links", type=int, nargs="*", default=None,
                       help="List of experiment IDs to link")
    p_test.add_argument("--items", dest="item_links", type=int, nargs="*", default=None,
                       help="List of item IDs to link")

    p_link = subparsers.add_parser("link-resources", 
                                  help="Link resources to an existing experiment")
    p_link.add_argument("--experiment-id", dest="experiment_id", type=int, required=True,
                       help="ID of the experiment to link resources to")
    p_link.add_argument("--experiments", dest="experiment_links", type=int, nargs="*", default=None,
                       help="List of experiment IDs to link")
    p_link.add_argument("--items", dest="item_links", type=int, nargs="*", default=None,
                       help="List of item IDs to link")

    args = parser.parse_args(argv)

    if args.command == "upload-training":
        return cmd_upload_training_results(
            run_label=args.run_label,
            title_prefix=args.title_prefix,
            category=args.category,
            team=args.team,
            experiment_links=args.experiment_links,
            item_links=args.item_links
        )
    elif args.command == "upload-test":
        return cmd_upload_test_results(
            run_label=args.run_label,
            title_prefix=args.title_prefix,
            category=args.category,
            team=args.team,
            experiment_links=args.experiment_links,
            item_links=args.item_links
        )
    elif args.command == "link-resources":
        return cmd_link_resources(
            experiment_id=args.experiment_id,
            experiment_links=args.experiment_links,
            item_links=args.item_links
        )
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
