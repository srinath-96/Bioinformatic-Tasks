"""
API client managers for variant annotation services.
"""

import requests
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.parse

from config.settings import settings, APIConfig

logger = logging.getLogger(__name__)

@dataclass
class AnnotationResult:
    """Result from an annotation API call."""
    variant_id: str
    source: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None

class APIClient:
    """Base class for annotation API clients."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting."""
        if self.config.rate_limit > 0:
            min_interval = 1.0 / self.config.rate_limit
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, **kwargs) -> requests.Response:
        """Make a rate-limited request with retry logic."""
        self._rate_limit()
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.get(
                    url, 
                    timeout=self.config.timeout,
                    **kwargs
                )
                
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"Rate limited by {self.config.name}. Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise
    
    def annotate_variant(self, variant_id: str, **kwargs) -> AnnotationResult:
        """Annotate a single variant. To be implemented by subclasses."""
        raise NotImplementedError

class MyVariantClient(APIClient):
    """Client for MyVariant.info API."""
    
    def annotate_variant(self, variant_id: str, **kwargs) -> AnnotationResult:
        """Annotate variant using MyVariant.info."""
        try:
            # Handle different variant ID formats
            if variant_id.startswith('rs'):
                # dbSNP ID - use directly
                myvariant_id = variant_id
            else:
                # Parse variant_id (chr:pos:ref>alt) and convert to MyVariant format
                parts = variant_id.split(':')
                if len(parts) == 3:
                    chrom, pos, ref_alt = parts
                    ref, alt = ref_alt.split('>')
                    
                    # Try simple chr:pos:ref>alt format first
                    myvariant_id = f"chr{chrom}:{pos}:{ref}>{alt}"
                else:
                    # Use as-is if format is unexpected
                    myvariant_id = variant_id
            
            url = f"{self.config.base_url}/variant/{myvariant_id}"
            
            params = {
                'fields': ','.join(settings.annotation_fields['myvariant'])
            }
            
            response = self._make_request(url, params=params)
            data = response.json()
            
            return AnnotationResult(
                variant_id=variant_id,
                source='myvariant',
                success=True,
                data=data
            )
            
        except Exception as e:
            # If first format fails, try alternative format  
            if not variant_id.startswith('rs') and ':' in variant_id:
                try:
                    parts = variant_id.split(':')
                    if len(parts) == 3:
                        chrom, pos, ref_alt = parts
                        ref, alt = ref_alt.split('>')
                        
                        # Try alternative format: chr#:g.pos:ref>alt
                        myvariant_id_alt = f"chr{chrom}:g.{pos}{ref}>{alt}"
                        url_alt = f"{self.config.base_url}/variant/{myvariant_id_alt}"
                        
                        response = self._make_request(url_alt, params={
                            'fields': ','.join(settings.annotation_fields['myvariant'])
                        })
                        data = response.json()
                        
                        return AnnotationResult(
                            variant_id=variant_id,
                            source='myvariant',
                            success=True,
                            data=data
                        )
                except:
                    pass  # Fall through to error handling
            
            logger.error(f"MyVariant annotation failed for {variant_id}: {e}")
            return AnnotationResult(
                variant_id=variant_id,
                source='myvariant',
                success=False,
                data={},
                error=str(e)
            )

class EnsemblClient(APIClient):
    """Client for Ensembl REST API."""
    
    def annotate_variant(self, variant_id: str, **kwargs) -> AnnotationResult:
        """Annotate variant using Ensembl VEP."""
        try:
            # Handle different variant ID formats
            if variant_id.startswith('rs') or variant_id.startswith('COSM'):
                # dbSNP or COSMIC ID - use ID endpoint
                import urllib.parse
                encoded_id = urllib.parse.quote(variant_id)
                url = f"{self.config.base_url}/vep/human/id/{encoded_id}"
            else:
                # Parse variant_id (chr:pos:ref>alt) and convert to genomic HGVS
                parts = variant_id.split(':')
                if len(parts) == 3:
                    chrom, pos, ref_alt = parts
                    ref, alt = ref_alt.split('>')
                    
                    # Format: chromosome:g.positionref>alt (e.g., 9:g.22125504G>C)
                    hgvs_notation = f"{chrom}:g.{pos}{ref}>{alt}"
                    
                    # URL encode the HGVS notation for the GET request
                    import urllib.parse
                    encoded_hgvs = urllib.parse.quote(hgvs_notation)
                    
                    url = f"{self.config.base_url}/vep/human/hgvs/{encoded_hgvs}"
                else:
                    # Fallback - assume it's already in correct format
                    import urllib.parse
                    encoded_id = urllib.parse.quote(variant_id)
                    url = f"{self.config.base_url}/vep/human/id/{encoded_id}"
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = self._make_request(url, headers=headers)
            result_data = response.json()
            
            return AnnotationResult(
                variant_id=variant_id,
                source='ensembl',
                success=True,
                data=result_data
            )
            
        except Exception as e:
            logger.error(f"Ensembl annotation failed for {variant_id}: {e}")
            return AnnotationResult(
                variant_id=variant_id,
                source='ensembl',
                success=False,
                data={},
                error=str(e)
            )

class ClinVarClient(APIClient):
    """Client for ClinVar API."""
    
    def annotate_variant(self, variant_id: str, **kwargs) -> AnnotationResult:
        """Annotate variant using ClinVar."""
        try:
            # Handle different variant ID formats
            if variant_id.startswith('rs'):
                # dbSNP ID - search by rsid
                search_term = f'"{variant_id}"[All Fields]'
            else:
                # Parse variant_id for genomic position search
                parts = variant_id.split(':')
                if len(parts) == 3:
                    chrom, pos, ref_alt = parts
                    ref, alt = ref_alt.split('>')
                    
                    # ClinVar search by position and alleles
                    search_term = f'{chrom}[chr] AND {pos}[pos] AND "{ref}"[ref] AND "{alt}"[alt]'
                else:
                    # Fallback search
                    search_term = f'"{variant_id}"[All Fields]'
            
            url = f"{self.config.base_url}/esearch.fcgi"
            
            params = {
                'db': 'clinvar',
                'term': search_term,
                'retmode': 'json',
                'retmax': 10
            }
            
            response = self._make_request(url, params=params)
            data = response.json()
            
            return AnnotationResult(
                variant_id=variant_id,
                source='clinvar',
                success=True,
                data=data
            )
            
        except Exception as e:
            logger.error(f"ClinVar annotation failed for {variant_id}: {e}")
            return AnnotationResult(
                variant_id=variant_id,
                source='clinvar',
                success=False,
                data={},
                error=str(e)
            )

class APIClientManager:
    """Manager for all annotation API clients with intelligent fallback."""
    
    def __init__(self):
        self.clients = {}
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize all configured API clients."""
        client_classes = {
            'myvariant': MyVariantClient,
            'ensembl': EnsemblClient,
            'clinvar': ClinVarClient
        }
        
        for api_config in settings.annotation_apis:
            if api_config.name in client_classes:
                self.clients[api_config.name] = client_classes[api_config.name](api_config)
                logger.info(f"Initialized {api_config.name} client")
    
    def annotate_variants(self, variant_ids: List[str]) -> Dict[str, List[AnnotationResult]]:
        """Annotate multiple variants using all available APIs."""
        results = {vid: [] for vid in variant_ids}
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=settings.max_concurrent_requests) as executor:
            futures = []
            
            for variant_id in variant_ids:
                for client_name, client in self.clients.items():
                    future = executor.submit(client.annotate_variant, variant_id)
                    futures.append((future, variant_id))
            
            # Collect results
            for future, variant_id in futures:
                try:
                    result = future.result(timeout=30)
                    results[variant_id].append(result)
                except Exception as e:
                    logger.error(f"Error getting annotation result for {variant_id}: {e}")
        
        return results
    
    def get_best_annotation(self, variant_id: str) -> Optional[AnnotationResult]:
        """Get the best available annotation for a variant."""
        for api_config in settings.annotation_apis:
            if api_config.name in self.clients:
                client = self.clients[api_config.name]
                result = client.annotate_variant(variant_id)
                
                if result.success:
                    logger.debug(f"Successfully annotated {variant_id} with {api_config.name}")
                    return result
                else:
                    logger.warning(f"Failed to annotate {variant_id} with {api_config.name}: {result.error}")
        
        logger.error(f"All annotation attempts failed for {variant_id}")
        return None 