#!/usr/bin/env python3
"""
Check existing experiment with tags to debug the issue
"""

import os
import elabapi_python
from elabapi_python import Configuration, ApiClient, ExperimentsApi
import json

def check_existing_experiment():
    """Check existing experiment with tags"""
    
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
        
        print("=== CHECKING EXISTING EXPERIMENT WITH TAGS ===\n")
        
        # Check experiment 176 (the one we created with tags)
        experiment_id = 176
        
        configuration = Configuration()
        configuration.api_key["api_key"] = api_key
        configuration.api_key_prefix["api_key"] = "Authorization"
        configuration.host = host_url
        configuration.verify_ssl = verify_ssl
        configuration.debug = True
        
        api_client = ApiClient(configuration)
        api_client.set_default_header(header_name="Authorization", header_value=api_key)
        experiments_api = ExperimentsApi(api_client)
        
        print(f"1. Checking experiment {experiment_id} with Python client:")
        
        try:
            exp = experiments_api.get_experiment(id=experiment_id)
            print(f"   ✅ Retrieved experiment {experiment_id}")
            print(f"   Title: {exp.title}")
            print(f"   Body: {exp.body}")
            print(f"   Category: {getattr(exp, 'category', 'N/A')}")
            print(f"   Team: {getattr(exp, 'team', 'N/A')}")
            
            # Check all attributes for tags
            print(f"\n   All attributes containing 'tag':")
            tag_attrs = []
            for attr in dir(exp):
                if not attr.startswith('_') and not callable(getattr(exp, attr)):
                    if "tag" in attr.lower():
                        value = getattr(exp, attr)
                        tag_attrs.append((attr, value))
                        print(f"     {attr}: {value}")
            
            if not tag_attrs:
                print(f"     No tag-related attributes found!")
                
            # Check if there are any other attributes that might contain tag info
            print(f"\n   All available attributes:")
            all_attrs = []
            for attr in dir(exp):
                if not attr.startswith('_') and not callable(getattr(exp, attr)):
                    value = getattr(exp, attr)
                    all_attrs.append((attr, value))
                    print(f"     {attr}: {value}")
                    
            # Look for any field that might contain the tag information
            print(f"\n   Looking for tag information in other fields:")
            for attr, value in all_attrs:
                if value and isinstance(value, str) and ("LodeSTAR" in value or "ML" in value):
                    print(f"     Potential tag info in {attr}: {value}")
                    
        except Exception as e:
            print(f"   ❌ Failed to get experiment {experiment_id}: {e}")
            import traceback
            traceback.print_exc()
            
        print("\n" + "="*60 + "\n")
        
        # Also check experiment 155 (the one from your JSON that definitely has tags)
        print(f"2. Checking experiment 155 (your working example):")
        
        try:
            exp = experiments_api.get_experiment(id=155)
            print(f"   ✅ Retrieved experiment 155")
            print(f"   Title: {exp.title}")
            print(f"   Body: {exp.body}")
            print(f"   Category: {getattr(exp, 'category', 'N/A')}")
            print(f"   Team: {getattr(exp, 'team', 'N/A')}")
            
            # Check all attributes for tags
            print(f"\n   All attributes containing 'tag':")
            tag_attrs = []
            for attr in dir(exp):
                if not attr.startswith('_') and not callable(getattr(exp, attr)):
                    if "tag" in attr.lower():
                        value = getattr(exp, attr)
                        tag_attrs.append((attr, value))
                        print(f"     {attr}: {value}")
            
            if not tag_attrs:
                print(f"     No tag-related attributes found!")
                
        except Exception as e:
            print(f"   ❌ Failed to get experiment 155: {e}")
            import traceback
            traceback.print_exc()
                
    except Exception as e:
        print(f"Error in setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_existing_experiment()
