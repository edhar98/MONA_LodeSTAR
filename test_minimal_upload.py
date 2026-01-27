#!/usr/bin/env python3
"""
Minimal test to replicate the working upload functionality
"""

import os
import elabapi_python
from elabapi_python import Configuration, ApiClient, ExperimentsApi, UploadsApi

def test_minimal_upload():
    """Test minimal upload functionality"""
    
    # Get environment variables
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
        
        # Try to create a minimal experiment (exactly like our working test)
        experiments_api = ExperimentsApi(api_client)
        
        test_body = {
            "title": "Minimal Upload Test",
            "body": "Testing minimal upload functionality",
            "template": 1
        }
        
        print(f"Attempting to create experiment with body: {test_body}")
        
        try:
            response = experiments_api.post_experiment_with_http_info(body=test_body)
            print(f"Success! Response:")
            print(f"Response type: {type(response)}")
            print(f"Response length: {len(response)}")
            
            # Extract experiment ID from Location header
            headers = response[2]  # Headers are at index 2
            location_header = headers.get('Location')
            if location_header:
                experiment_id = location_header.split('/')[-1]
                print(f"Experiment ID: {experiment_id}")
                print(f"Location: {location_header}")
                
                # Now try to upload a test file
                print(f"\n--- Testing file upload to experiment {experiment_id} ---")
                
                uploads_api = UploadsApi(api_client)
                
                # Create a simple test file
                test_file_content = "This is a test file for minimal upload test"
                test_filename = "minimal_test.txt"
                
                with open(test_filename, 'w') as f:
                    f.write(test_file_content)
                
                try:
                    upload_response = uploads_api.post_upload(
                        id=int(experiment_id), 
                        file=test_filename, 
                        entity_type="experiments"
                    )
                    print(f"File upload successful!")
                    print(f"Upload response: {upload_response}")
                except Exception as upload_e:
                    print(f"File upload failed: {upload_e}")
                    if hasattr(upload_e, 'body'):
                        print(f"Upload error body: {upload_e.body}")
                
                # Clean up test file
                if os.path.exists(test_filename):
                    os.remove(test_filename)
                
            else:
                print("No Location header found in response")
                print(f"All headers: {dict(headers)}")
                
        except Exception as e:
            print(f"Error creating experiment: {e}")
            print(f"Error type: {type(e)}")
            
            # Try to get more details about the error
            if hasattr(e, 'body'):
                print(f"Error body: {e.body}")
            if hasattr(e, 'status'):
                print(f"Error status: {e.status}")
            if hasattr(e, 'reason'):
                print(f"Error reason: {e.reason}")
                
    except Exception as e:
        print(f"Error in setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal_upload()
