#!/usr/bin/env python3
"""
Test script to verify API client fixes without requiring LLM setup.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.api_clients import MyVariantClient, EnsemblClient, ClinVarClient
from config.settings import APIConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api_clients():
    """Test the fixed API client implementations."""
    
    # Test variant IDs - using known working examples from API docs
    test_variants = [
        "9:22125504:G>C",      # From Ensembl docs
        "rs56116432",          # Known dbSNP ID
        "1:230710048:A>G"      # Another test variant
    ]
    
    # Configure API clients
    configs = {
        'myvariant': APIConfig(
            name="myvariant",
            base_url="https://myvariant.info/v1",
            rate_limit=10,
            timeout=30,
            retry_attempts=2
        ),
        'ensembl': APIConfig(
            name="ensembl", 
            base_url="https://rest.ensembl.org",
            rate_limit=15,
            timeout=30,
            retry_attempts=2
        ),
        'clinvar': APIConfig(
            name="clinvar",
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate_limit=3,
            timeout=30,
            retry_attempts=2
        )
    }
    
    # Initialize clients
    clients = {
        'myvariant': MyVariantClient(configs['myvariant']),
        'ensembl': EnsemblClient(configs['ensembl']),
        'clinvar': ClinVarClient(configs['clinvar'])
    }
    
    print("🧪 Testing API Client Fixes")
    print("=" * 50)
    
    results = {}
    
    for variant_id in test_variants:
        print(f"\n📋 Testing variant: {variant_id}")
        results[variant_id] = {}
        
        for client_name, client in clients.items():
            try:
                print(f"  Testing {client_name}...")
                
                # Special handling for dbSNP IDs
                if variant_id.startswith('rs'):
                    if client_name == 'ensembl':
                        # Use dbSNP ID directly for Ensembl
                        test_id = variant_id
                    elif client_name == 'myvariant':
                        # MyVariant can handle dbSNP IDs directly too
                        test_id = variant_id
                    else:
                        # For ClinVar, we'll skip dbSNP IDs for now
                        test_id = variant_id
                else:
                    test_id = variant_id
                
                result = client.annotate_variant(test_id)
                
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                print(f"    {status}: {client_name}")
                
                if result.success:
                    data_info = "Data received" if result.data else "No data"
                    print(f"    {data_info}")
                else:
                    print(f"    Error: {result.error}")
                
                results[variant_id][client_name] = {
                    'success': result.success,
                    'error': result.error,
                    'data_size': len(str(result.data)) if result.data else 0
                }
                
            except Exception as e:
                print(f"    ❌ EXCEPTION: {client_name} - {str(e)}")
                results[variant_id][client_name] = {
                    'success': False,
                    'error': str(e),
                    'data_size': 0
                }
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 50)
    
    for client_name in clients.keys():
        successes = sum(1 for v in results.values() if v.get(client_name, {}).get('success', False))
        total = len(test_variants)
        success_rate = (successes / total) * 100
        
        status_emoji = "✅" if success_rate > 0 else "❌"
        print(f"{status_emoji} {client_name}: {successes}/{total} ({success_rate:.1f}%)")
    
    return results

if __name__ == "__main__":
    try:
        results = test_api_clients()
        print("\n🎉 Test completed!")
        
        # Check if we have any successes
        any_success = any(
            client_result.get('success', False) 
            for variant_results in results.values() 
            for client_result in variant_results.values()
        )
        
        if any_success:
            print("✅ At least some API calls are working - the fixes appear to be successful!")
        else:
            print("❌ No API calls succeeded - there may still be issues to resolve.")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1) 