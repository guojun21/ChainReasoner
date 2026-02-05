#!/usr/bin/env python3
"""
Simple test script to verify configuration and API connections.
Tests LLM API and basic functionality without requiring Neo4j.
"""



import json
from pathlib import Path

import requests
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


def load_config():
    """Load configuration from config.yaml."""
    with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_llm_api(config):
    """Test the LLM API connection."""
    print("\n" + "="*50)
    print("Testing LLM API Connection...")
    print("="*50)
    
    base_model = config.get("base_model", {})
    api_url = base_model.get("api_url")
    api_key = base_model.get("api_key")
    model_id = base_model.get("model_id")
    
    print(f"API URL: {api_url}")
    print(f"Model ID: {model_id}")
    print(f"API Key: {api_key[:10]}...{api_key[-10:]}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Please respond with 'API connection successful!'"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        print("\nSending request to API...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print(f"\nLLM Response: {content}")
            return True
        else:
            print("\nWarning: Unexpected response format")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to API: {e}")
        return False


def test_brave_api(config):
    """Test Brave Search API connection."""
    print("\n" + "="*50)
    print("Testing Brave Search API Connection...")
    print("="*50)
    
    api_keys = config.get("api_keys", {})
    brave_key = api_keys.get("brave_api_key")

    if not brave_key:
        print("Brave API key is missing in configs/config.yaml")
        return False
    
    print(f"Brave API Key: {brave_key[:10]}...{brave_key[-10:]}")
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_key
    }
    
    params = {
        "q": "test query",
        "count": 1
    }
    
    try:
        print("\nSending request to Brave Search API...")
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"\nStatus Code: {response.status_code}")
        total_results = len(result.get("web", {}).get("results", []))
        print(f"Total Results: {total_results}")
        
        if result.get("web", {}).get("results"):
            print(f"First Result: {result['web']['results'][0].get('title', '')}")
            return True
        else:
            print("\nWarning: Unexpected response format")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\nError connecting to Brave API: {e}")
        return False


def test_basic_components():
    """Test basic Python components."""
    print("\n" + "="*50)
    print("Testing Basic Components...")
    print("="*50)
    
    all_passed = True
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"✗ PyYAML import error: {e}")
        all_passed = False
    
    try:
        import requests
        print("✓ Requests imported successfully")
    except ImportError as e:
        print(f"✗ Requests import error: {e}")
        all_passed = False
    
    try:
        import neo4j
        print("✓ Neo4j driver imported successfully")
    except ImportError as e:
        print(f"✗ Neo4j import error: {e}")
        all_passed = False
    
    try:
        import torch
        print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        print(f"⚠ PyTorch import error (optional): {e}")
    
    try:
        import transformers
        print(f"✓ Transformers imported successfully (version: {transformers.__version__})")
    except ImportError as e:
        print(f"⚠ Transformers import error (optional): {e}")
    
    try:
        from rank_bm25 import BM25Okapi
        print("✓ BM25 imported successfully")
    except ImportError as e:
        print(f"⚠ BM25 import error (optional): {e}")
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*50)
    print("MultiHop Agent - Configuration Test")
    print("="*50)
    
    config = load_config()
    print("\n✓ Configuration loaded successfully")
    
    results = {}
    
    results["basic_components"] = test_basic_components()
    results["llm_api"] = test_llm_api(config)
    results["brave_api"] = test_brave_api(config)
    
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNote: Neo4j database is still required for full functionality.")
        print("Please install Neo4j and update configs/config.yaml with correct credentials.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)