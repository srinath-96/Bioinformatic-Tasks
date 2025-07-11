"""
Annotator agent for the variant annotation pipeline.
"""

from crewai import Agent
from typing import Dict, Any, List
import pandas as pd
import logging
import time

from config.agents_config import AgentsConfig
from config.settings import settings
from utils.api_clients import APIClientManager
from utils.data_handlers import DataHandler

logger = logging.getLogger(__name__)

class AnnotatorAgent:
    """Agent responsible for intelligent variant annotation with fallback strategies."""
    
    def __init__(self):
        self.config = AgentsConfig.ANNOTATOR
        self.api_manager = APIClientManager()
        self.data_handler = DataHandler()
        
        # Get LLM configuration
        self.llm = AgentsConfig.get_configured_llm()
        
        # Create CrewAI agent with LLM
        self.agent = Agent(
            role=self.config.role,
            goal=self.config.goal,
            backstory=self.config.backstory,
            verbose=self.config.verbose,
            allow_delegation=self.config.allow_delegation,
            max_iter=self.config.max_iter,
            llm=self.llm
        )
        
        # Performance tracking
        self.annotation_stats = {
            'total_variants': 0,
            'successful_annotations': 0,
            'failed_annotations': 0,
            'processing_time': 0
        }
    
    def annotate_variants(self, variants_df: pd.DataFrame) -> Dict[str, Any]:
        """Annotate all variants in the DataFrame using intelligent strategies."""
        start_time = time.time()
        logger.info(f"AnnotatorAgent: Starting annotation of {len(variants_df)} variants")
        
        self.annotation_stats['total_variants'] = len(variants_df)
        
        result = {
            'success': False,
            'annotated_data': None,
            'annotation_results': {},
            'failed_variants': [],
            'stats': {},
            'errors': []
        }
        
        try:
            # Get variant IDs to annotate
            variant_ids = variants_df['variant_id'].tolist()
            
            # Batch annotate variants
            annotations = self._batch_annotate(variant_ids)
            
            # Merge annotations with variant data
            annotated_df = self.data_handler.merge_variant_annotations(
                variants_df, annotations
            )
            
            # Generate final statistics
            self.annotation_stats['processing_time'] = time.time() - start_time
            final_stats = self._generate_annotation_stats(annotated_df)
            
            result.update({
                'success': True,
                'annotated_data': annotated_df,
                'annotation_results': annotations,
                'stats': final_stats
            })
            
            logger.info(f"AnnotatorAgent: Successfully annotated {len(annotated_df)} variants in {self.annotation_stats['processing_time']:.2f}s")
            
        except Exception as e:
            error_msg = f"Error during variant annotation: {str(e)}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
        
        return result
    
    def _batch_annotate(self, variant_ids: List[str]) -> Dict[str, List]:
        """Annotate variants in batches."""
        all_results = {}
        
        if not variant_ids:
            return all_results
        
        # Process in batches to respect API limits
        batch_size = settings.max_variants_per_batch
        batches = [variant_ids[i:i + batch_size] for i in range(0, len(variant_ids), batch_size)]
        
        logger.info(f"Processing {len(variant_ids)} variants in {len(batches)} batches")
        
        for batch_idx, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_idx}/{len(batches)} ({len(batch)} variants)")
            
            # Annotate batch with fallback strategy
            batch_results = self._annotate_batch_with_fallback(batch)
            all_results.update(batch_results)
            
            # Brief pause between batches
            if batch_idx < len(batches):
                time.sleep(1)
        
        return all_results
    
    def _annotate_batch_with_fallback(self, variant_ids: List[str]) -> Dict[str, List]:
        """Annotate a batch of variants with fallback strategies."""
        results = {}
        
        # Try concurrent annotation with all APIs
        try:
            concurrent_results = self.api_manager.annotate_variants(variant_ids)
            
            # Check success rate
            total_calls = sum(len(res) for res in concurrent_results.values())
            successful_calls = sum(
                sum(1 for r in res if r.success) for res in concurrent_results.values()
            )
            
            success_rate = successful_calls / total_calls if total_calls > 0 else 0
            
            if success_rate >= 0.5:  # At least 50% success rate
                logger.debug(f"Concurrent annotation succeeded with {success_rate:.1%} success rate")
                return concurrent_results
            else:
                logger.warning(f"Concurrent annotation had low success rate ({success_rate:.1%}), trying fallback")
        
        except Exception as e:
            logger.warning(f"Concurrent annotation failed: {e}, trying fallback")
        
        # Fallback: Sequential annotation with best API
        logger.info("Falling back to sequential annotation with best available API")
        
        for variant_id in variant_ids:
            best_result = self.api_manager.get_best_annotation(variant_id)
            if best_result:
                results[variant_id] = [best_result]
                self.annotation_stats['successful_annotations'] += 1
            else:
                results[variant_id] = []
                self.annotation_stats['failed_annotations'] += 1
                logger.warning(f"Failed to annotate variant {variant_id} with any API")
        
        return results
    
    def _generate_annotation_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive annotation statistics."""
        stats = {
            'total_variants_processed': len(df),
            'annotation_success_rate': df['annotation_success_rate'].mean() if len(df) > 0 else 0,
            'processing_time_seconds': self.annotation_stats['processing_time'],
            'variants_per_second': len(df) / self.annotation_stats['processing_time'] if self.annotation_stats['processing_time'] > 0 else 0,
        }
        
        # API usage statistics
        if len(df) > 0 and 'annotation_sources' in df.columns:
            all_sources = []
            for sources_str in df['annotation_sources'].dropna():
                all_sources.extend(sources_str.split(','))
            
            api_usage = {}
            for source in all_sources:
                api_usage[source] = api_usage.get(source, 0) + 1
            
            stats['api_usage'] = api_usage
            stats['most_successful_api'] = max(api_usage.items(), key=lambda x: x[1])[0] if api_usage else None
        
        return stats 