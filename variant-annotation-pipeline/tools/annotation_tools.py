"""
Annotation-specific tools for the annotator agent.
"""

from crewai.tools import BaseTool
from typing import Dict, Any, List
import time
import logging

logger = logging.getLogger(__name__)

class AnnotationRetryTool(BaseTool):
    """Tool for implementing smart retry logic for failed annotations."""
    
    name: str = "Annotation Retry"
    description: str = "Implements intelligent retry strategies for failed variant annotations."
    
    def _run(self, failed_variants: List[str], retry_strategy: str = "exponential") -> Dict[str, Any]:
        """Retry annotation for failed variants."""
        from utils.api_clients import APIClientManager
        
        retry_results = {
            'successful_retries': [],
            'failed_retries': [],
            'retry_attempts': 0
        }
        
        if not failed_variants:
            return retry_results
        
        api_manager = APIClientManager()
        max_retries = 3
        
        for attempt in range(max_retries):
            retry_results['retry_attempts'] += 1
            
            # Calculate delay based on strategy
            if retry_strategy == "exponential":
                delay = 2 ** attempt
            elif retry_strategy == "linear":
                delay = attempt + 1
            else:  # immediate
                delay = 0
            
            if delay > 0:
                time.sleep(delay)
            
            # Retry failed variants
            still_failing = []
            for variant_id in failed_variants:
                result = api_manager.get_best_annotation(variant_id)
                if result and result.success:
                    retry_results['successful_retries'].append(variant_id)
                else:
                    still_failing.append(variant_id)
            
            failed_variants = still_failing
            
            if not failed_variants:
                break
        
        retry_results['failed_retries'] = failed_variants
        return retry_results

class AnnotationValidatorTool(BaseTool):
    """Tool for validating annotation data quality and completeness."""
    
    name: str = "Annotation Validator"
    description: str = "Validates the quality and completeness of variant annotation data."
    
    def _run(self, annotation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate annotation data."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'completeness_score': 0.0
        }
        
        try:
            # Check if annotation_data has expected structure
            if not isinstance(annotation_data, dict):
                validation_results['errors'].append("Annotation data must be a dictionary")
                validation_results['is_valid'] = False
                return validation_results
            
            # Check for required fields
            required_fields = ['variant_id', 'annotation_sources', 'annotation_success_rate']
            missing_fields = []
            
            for field in required_fields:
                if field not in annotation_data:
                    missing_fields.append(field)
            
            if missing_fields:
                validation_results['errors'].append(f"Missing required fields: {missing_fields}")
                validation_results['is_valid'] = False
            
            # Calculate completeness score
            success_rate = annotation_data.get('annotation_success_rate', 0)
            validation_results['completeness_score'] = success_rate
            
            # Generate warnings based on quality
            if success_rate < 0.3:
                validation_results['warnings'].append("Low annotation completeness (< 30%)")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
