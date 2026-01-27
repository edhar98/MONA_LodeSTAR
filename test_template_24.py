#!/usr/bin/env python3
"""
Test script to check template ID 24 compatibility
"""

import os
import elabapi_python
from elabapi_python import Configuration, ApiClient, ExperimentsApi

def test_template_24():
    """Test template ID 24 compatibility"""
    
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
        
        # Test template ID 24 with different field combinations
        experiments_api = ExperimentsApi(api_client)
        
        print("\n=== Testing Template ID 24 ===")
        
        # Test 1: Minimal fields only (like template 1)
        print("Test 1: Minimal fields only")
        test_body_minimal = {
            "title": "Template 24 Test - Minimal",
            "body": "Testing template 24 with minimal fields",
            "template": 24
        }
        
        try:
            response = experiments_api.post_experiment_with_http_info(body=test_body_minimal)
            print(f"✅ Success with minimal fields!")
            print(f"Response type: {type(response)}")
            print(f"Response length: {len(response)}")
            
            # Extract experiment ID
            location_header = response[2].get("Location")
            if location_header:
                experiment_id = location_header.split("/")[-1]
                print(f"Experiment ID: {experiment_id}")
                return  # Success, no need to test more
                
        except Exception as e:
            print(f"❌ Failed with minimal fields: {e}")
            if hasattr(e, 'body'):
                print(f"Error body: {e.body}")
            if hasattr(e, 'status'):
                print(f"Error status: {e.status}")
            if hasattr(e, 'reason'):
                print(f"Error reason: {e.reason}")
        
        # Test 2: With just title and template (no body)
        print("\nTest 2: Title and template only")
        test_body_title_only = {
            "title": "Template 24 Test - Title Only",
            "template": 24
        }
        
        try:
            response = experiments_api.post_experiment_with_http_info(body=test_body_title_only)
            print(f"✅ Success with title only!")
            print(f"Response type: {type(response)}")
            print(f"Response length: {len(response)}")
            
            # Extract experiment ID
            location_header = response[2].get("Location")
            if location_header:
                experiment_id = location_header.split("/")[-1]
                print(f"Experiment ID: {experiment_id}")
                return  # Success
                
        except Exception as e:
            print(f"❌ Failed with title only: {e}")
            if hasattr(e, 'body'):
                print(f"Error body: {e.body}")
            if hasattr(e, 'status'):
                print(f"Error status: {e.status}")
            if hasattr(e, 'reason'):
                print(f"Error reason: {e.reason}")
        
        # Test 3: Try template ID 1 again to confirm it still works
        print("\nTest 3: Template ID 1 (should work)")
        test_body_template_1 = {
            "title": "Template 1 Test - Confirmation",
            "body": "Testing template 1 still works",
            "template": 1
        }
        
        try:
            response = experiments_api.post_experiment_with_http_info(body=test_body_template_1)
            print(f"✅ Template 1 still works!")
            print(f"Response type: {type(response)}")
            print(f"Response length: {len(response)}")
            
            # Extract experiment ID
            location_header = response[2].get("Location")
            if location_header:
                experiment_id = location_header.split("/")[-1]
                print(f"Experiment ID: {experiment_id}")
                
        except Exception as e:
            print(f"❌ Template 1 failed: {e}")
            if hasattr(e, 'body'):
                print(f"Error body: {e.body}")
            if hasattr(e, 'status'):
                print(f"Error status: {e.status}")
            if hasattr(e, 'reason'):
                print(f"Error reason: {e.reason}")
        
        print("\n❌ Template ID 24 tests failed. May need to investigate template requirements.")
                
    except Exception as e:
        print(f"Error in setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_template_24()
