#!/usr/bin/env python3
"""
Test script to check which metadata fields template ID 24 supports
"""

import os
import elabapi_python
from elabapi_python import Configuration, ApiClient, ExperimentsApi

def test_template_24_metadata():
    """Test template ID 24 metadata support"""
    
    # Get environment variables
    host_url = os.environ.get("ELAB_HOST_URL")
    api_key = os.environ.get("ELAB_API_KEY")
    verify_ssl = os.environ.get("ELAB_VERIFY_SSL", "false").strip().lower() == "true"
    
    if not host_url or not api_key:
        print("Missing environment variables")
        return
    
    try:
        # Fix the host URL
        if host_url.endswith('/'):
            host_url = host_url.rstrip('/')
        
        # Create configuration
        configuration = Configuration()
        configuration.api_key["api_key"] = api_key
        configuration.api_key_prefix["api_key"] = "Authorization"
        configuration.host = host_url
        configuration.verify_ssl = verify_ssl
        configuration.debug = False  # Disable debug for cleaner output
        
        # Create API client
        api_client = ApiClient(configuration)
        api_client.set_default_header(header_name="Authorization", header_value=api_key)
        experiments_api = ExperimentsApi(api_client)
        
        print("=== Testing Template ID 24 Metadata Support ===\n")
        
        # Test 1: Minimal fields (we know this works)
        print("✅ Test 1: Minimal fields (title + body + template)")
        test_body = {
            "title": "Template 24 Metadata Test - Minimal",
            "body": "Testing minimal fields",
            "template": 24
        }
        
        try:
            response = experiments_api.post_experiment_with_http_info(body=test_body)
            experiment_id = response[2].get("Location").split("/")[-1]
            print(f"   Success! Experiment ID: {experiment_id}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return
        
        # Test 2: Add tags
        print("\n✅ Test 2: Add tags")
        test_body = {
            "title": "Template 24 Metadata Test - With Tags",
            "body": "Testing with tags",
            "template": 24,
            "tags": "LodeSTAR|Test Run|ML"
        }
        
        try:
            response = experiments_api.post_experiment_with_http_info(body=test_body)
            experiment_id = response[2].get("Location").split("/")[-1]
            print(f"   Success! Experiment ID: {experiment_id}")
        except Exception as e:
            print(f"   ❌ Tags failed: {e}")
            test_body.pop("tags")  # Remove tags for next test
        
        # Test 3: Add category
        print("\n✅ Test 3: Add category")
        test_body["category"] = 5
        
        try:
            response = experiments_api.post_experiment_with_http_info(body=test_body)
            experiment_id = response[2].get("Location").split("/")[-1]
            print(f"   Success! Experiment ID: {experiment_id}")
        except Exception as e:
            print(f"   ❌ Category failed: {e}")
            test_body.pop("category")  # Remove category for next test
        
        # Test 4: Add team
        print("\n✅ Test 4: Add team")
        test_body["team"] = 1
        
        try:
            response = experiments_api.post_experiment_with_http_info(body=test_body)
            experiment_id = response[2].get("Location").split("/")[-1]
            print(f"   Success! Experiment ID: {experiment_id}")
        except Exception as e:
            print(f"   ❌ Team failed: {e}")
            test_body.pop("team")  # Remove team for next test
        
        # Test 5: Try all fields together
        print("\n✅ Test 5: All fields together")
        test_body_all = {
            "title": "Template 24 Metadata Test - All Fields",
            "body": "Testing all metadata fields",
            "template": 24,
            "tags": "LodeSTAR|Test Run|ML",
            "category": 5,
            "team": 1
        }
        
        try:
            response = experiments_api.post_experiment_with_http_info(body=test_body_all)
            experiment_id = response[2].get("Location").split("/")[-1]
            print(f"   Success! Experiment ID: {experiment_id}")
            print(f"   🎉 Template 24 supports all metadata fields!")
        except Exception as e:
            print(f"   ❌ All fields failed: {e}")
            print(f"   📝 Template 24 has limited metadata support")
        
        print(f"\n=== Summary ===")
        print(f"Template ID 24: {'✅ Working' if 'template' in test_body else '❌ Failed'}")
        print(f"Tags support: {'✅ Yes' if 'tags' in test_body else '❌ No'}")
        print(f"Category support: {'✅ Yes' if 'category' in test_body else '❌ No'}")
        print(f"Team support: {'✅ Yes' if 'team' in test_body else '❌ No'}")
                
    except Exception as e:
        print(f"Error in setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_template_24_metadata()
