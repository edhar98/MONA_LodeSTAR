#!/usr/bin/env python3
"""
Debug script to replicate the CLI upload-test-run functionality
"""

import os
import tarfile
import elabapi_python
from elabapi_python import Configuration, ApiClient, ExperimentsApi, UploadsApi

def archive_directories(directories, run_label):
    """Replicate the archive function from CLI"""
    archives = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        archive_name = f"{run_label}_{os.path.basename(directory)}.tar.gz"
        with tarfile.open(archive_name, "w:gz") as tar:
            tar.add(directory, arcname=os.path.basename(directory))
        archives.append(archive_name)
    return archives

def debug_upload_test_run():
    """Debug the upload-test-run functionality"""
    
    print("=== Debug Upload Test Run ===")
    
    # Test parameters
    run_label = "debug_test_run"
    title_prefix = "Debug Test Run"
    
    print(f"Run label: {run_label}")
    print(f"Title prefix: {title_prefix}")
    
    # Check directories
    test_directories = ["logs", "detection_results"]
    existing_dirs = [d for d in test_directories if os.path.isdir(d)]
    print(f"Existing directories: {existing_dirs}")
    
    if not existing_dirs:
        print("No directories found to archive")
        return
    
    # Check for test results summary
    additional_files = []
    test_summary = "test_results_summary.yaml"
    if os.path.exists(test_summary):
        additional_files.append(test_summary)
        print(f"Found test results summary: {test_summary}")
    
    # Create archives
    print("Creating archives...")
    archives = archive_directories(existing_dirs, run_label)
    print(f"Archives created: {archives}")
    
    if not archives:
        print("No archives created")
        return
    
    # Build API client (exactly like CLI)
    print("\n=== Building API Client ===")
    host_url = os.environ.get("ELAB_HOST_URL")
    api_key = os.environ.get("ELAB_API_KEY")
    verify_ssl = os.environ.get("ELAB_VERIFY_SSL", "false").strip().lower() == "true"
    
    print(f"Host URL: {host_url}")
    print(f"API Key: {api_key[:10]}..." if api_key else "None")
    print(f"Verify SSL: {verify_ssl}")
    
    if not host_url or not api_key:
        print("Missing environment variables")
        return
    
    try:
        # Fix the host URL - remove trailing slash to avoid double slashes
        if host_url.endswith('/'):
            host_url = host_url.rstrip('/')
        
        print(f"Fixed Host URL: {host_url}")
        
        # Create configuration
        configuration = Configuration()
        configuration.api_key["api_key"] = api_key
        configuration.api_key_prefix["api_key"] = "Authorization"
        configuration.host = host_url
        configuration.verify_ssl = verify_ssl
        configuration.debug = True  # Enable debug mode
        
        print(f"Configuration host: {configuration.host}")
        
        # Create API client
        api_client = ApiClient(configuration)
        api_client.set_default_header(header_name="Authorization", header_value=api_key)
        
        print("API client created successfully")
        
        # Create experiment (exactly like CLI)
        print("\n=== Creating Experiment ===")
        experiments_api = ExperimentsApi(api_client)
        
        title = f"{title_prefix} {run_label}".strip()
        body = f"Test run results from test_single_particle.py: {', '.join(existing_dirs)}"
        if additional_files:
            body += f"\nAdditional files: {', '.join(additional_files)}"
        
        # Prepare experiment body with required fields
        experiment_body = {"title": title, "body": body, "template": 1}
        
        print(f"Experiment body: {experiment_body}")
        
        try:
            exp_resp = experiments_api.post_experiment_with_http_info(body=experiment_body)
            print(f"Experiment created successfully!")
            print(f"Response type: {type(exp_resp)}")
            print(f"Response length: {len(exp_resp)}")
            
            # Extract experiment ID
            location_header = exp_resp[2].get("Location")
            if location_header:
                experiment_id = int(location_header.split("/")[-1])
                print(f"Experiment ID: {experiment_id}")
                
                # Upload files
                print(f"\n=== Uploading Files ===")
                uploads_api = UploadsApi(api_client)
                
                # Upload archives
                for archive in archives:
                    try:
                        print(f"Uploading {archive}...")
                        uploads_api.post_upload(
                            id=experiment_id, 
                            file=archive, 
                            entity_type="experiments"
                        )
                        print(f"Successfully uploaded {archive}")
                    except Exception as exc:
                        print(f"Failed to upload {archive}: {exc}")
                        return
                
                # Upload additional files
                for file_path in additional_files:
                    try:
                        print(f"Uploading {file_path}...")
                        uploads_api.post_upload(
                            id=experiment_id, 
                            file=file_path, 
                            entity_type="experiments"
                        )
                        print(f"Successfully uploaded {file_path}")
                    except Exception as exc:
                        print(f"Failed to upload {file_path}: {exc}")
                        return
                
                print(f"\n=== Success! ===")
                print(f"Experiment ID: {experiment_id}")
                print(f"Uploaded: {archives + additional_files}")
                
            else:
                print("No Location header found")
                
        except Exception as exc:
            print(f"Failed to create experiment: {exc}")
            if hasattr(exc, 'body'):
                print(f"Error body: {exc.body}")
            if hasattr(exc, 'status'):
                print(f"Error status: {exc.status}")
            if hasattr(exc, 'reason'):
                print(f"Error reason: {exc.reason}")
            import traceback
            traceback.print_exc()
                
    except Exception as e:
        print(f"Error in setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_upload_test_run()
