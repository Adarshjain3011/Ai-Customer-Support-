"""
Advanced A/B Testing System for AI Support Optimization
"""
import random
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
import statistics

class TestType(str, Enum):
    RESPONSE_STYLE = "response_style"
    ESCALATION_THRESHOLD = "escalation_threshold"
    RESPONSE_LENGTH = "response_length"
    INTENT_CLASSIFICATION = "intent_classification"
    CACHE_STRATEGY = "cache_strategy"

class VariantType(str, Enum):
    CONTROL = "control"
    TREATMENT_A = "treatment_a"
    TREATMENT_B = "treatment_b"
    TREATMENT_C = "treatment_c"

@dataclass
class ABTestConfig:
    """Configuration for an A/B test"""
    test_id: str
    test_name: str
    test_type: TestType
    description: str
    start_date: datetime
    end_date: Optional[datetime] = None
    coverage_percentage: float = 0.1  # 10% of users
    min_sample_size: int = 100
    confidence_level: float = 0.95
    variants: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TestResult:
    """Result of an A/B test variant"""
    variant: str
    sample_size: int = 0
    success_count: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    avg_satisfaction: float = 0.0
    escalation_rate: float = 0.0
    error_rate: float = 0.0
    conversion_rate: float = 0.0

class ABTestManager:
    """Advanced A/B testing manager for support system optimization"""
    
    def __init__(self, config):
        self.config = config
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, Dict[str, TestResult]] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> test_id -> variant
        self.conversion_events: List[Dict[str, Any]] = []
        
        # Initialize default tests
        self._setup_default_tests()
    
    def _setup_default_tests(self):
        """Setup default A/B tests"""
        default_tests = [
            ABTestConfig(
                test_id="response_style_001",
                test_name="Response Style Optimization",
                test_type=TestType.RESPONSE_STYLE,
                description="Test different response styles for user satisfaction",
                start_date=datetime.now(),
                coverage_percentage=0.15,
                variants={
                    "control": {"style": "formal", "tone": "professional"},
                    "treatment_a": {"style": "casual", "tone": "friendly"},
                    "treatment_b": {"style": "empathetic", "tone": "caring"},
                    "treatment_c": {"style": "technical", "tone": "precise"}
                }
            ),
            ABTestConfig(
                test_id="escalation_threshold_001",
                test_name="Escalation Threshold Optimization",
                test_type=TestType.ESCALATION_THRESHOLD,
                description="Test different escalation thresholds for optimal routing",
                start_date=datetime.now(),
                coverage_percentage=0.1,
                variants={
                    "control": {"threshold": 0.8},
                    "treatment_a": {"threshold": 0.7},
                    "treatment_b": {"threshold": 0.9},
                    "treatment_c": {"threshold": 0.75}
                }
            ),
            ABTestConfig(
                test_id="response_length_001",
                test_name="Response Length Optimization",
                test_type=TestType.RESPONSE_LENGTH,
                description="Test optimal response length for user engagement",
                start_date=datetime.now(),
                coverage_percentage=0.1,
                variants={
                    "control": {"length": "concise", "max_words": 50},
                    "treatment_a": {"length": "detailed", "max_words": 100},
                    "treatment_b": {"length": "step_by_step", "max_words": 150},
                    "treatment_c": {"length": "minimal", "max_words": 25}
                }
            )
        ]
        
        for test in default_tests:
            self.add_test(test)
    
    def add_test(self, test_config: ABTestConfig):
        """Add a new A/B test"""
        self.active_tests[test_config.test_id] = test_config
        self.test_results[test_config.test_id] = {}
        
        # Initialize result tracking for each variant
        for variant_name in test_config.variants.keys():
            self.test_results[test_config.test_id][variant_name] = TestResult(variant=variant_name)
    
    def get_user_variant(self, user_id: str, test_id: str) -> Optional[str]:
        """Get the variant assigned to a user for a specific test"""
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        if not test.is_active:
            return None
        
        # Check if user is already assigned
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        
        if test_id not in self.user_assignments[user_id]:
            # Determine if user should be included in test
            if self._should_include_user(user_id, test):
                variant = self._assign_variant(user_id, test)
                self.user_assignments[user_id][test_id] = variant
            else:
                return None
        
        return self.user_assignments[user_id].get(test_id)
    
    def _should_include_user(self, user_id: str, test: ABTestConfig) -> bool:
        """Determine if a user should be included in the test"""
        # Use consistent hashing for deterministic assignment
        user_hash = int(hashlib.md5(f"{user_id}_{test.test_id}".encode()).hexdigest(), 16)
        return (user_hash % 100) < (test.coverage_percentage * 100)
    
    def _assign_variant(self, user_id: str, test: ABTestConfig) -> str:
        """Assign a variant to a user"""
        variants = list(test.variants.keys())
        
        # Use user_id for consistent variant assignment
        user_hash = int(hashlib.md5(f"{user_id}_{test.test_id}".encode()).hexdigest(), 16)
        variant_index = user_hash % len(variants)
        
        return variants[variant_index]
    
    def track_conversion(self, user_id: str, test_id: str, event_type: str, 
                        value: Union[float, int, str], metadata: Dict[str, Any] = None):
        """Track a conversion event for A/B testing"""
        if test_id not in self.active_tests:
            return
        
        variant = self.get_user_variant(user_id, test_id)
        if not variant:
            return
        
        conversion_event = {
            "user_id": user_id,
            "test_id": test_id,
            "variant": variant,
            "event_type": event_type,
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversion_events.append(conversion_event)
        
        # Update test results
        self._update_test_results(test_id, variant, event_type, value)
    
    def _update_test_results(self, test_id: str, variant: str, event_type: str, value: Union[float, int, str]):
        """Update test results based on conversion events"""
        if test_id not in self.test_results or variant not in self.test_results[test_id]:
            return
        
        result = self.test_results[test_id][variant]
        
        if event_type == "request":
            result.total_requests += 1
        elif event_type == "success":
            result.success_count += 1
        elif event_type == "response_time":
            # Update average response time
            if result.sample_size == 0:
                result.avg_response_time = value
            else:
                result.avg_response_time = (result.avg_response_time * result.sample_size + value) / (result.sample_size + 1)
            result.sample_size += 1
        elif event_type == "satisfaction":
            # Update average satisfaction
            if result.sample_size == 0:
                result.avg_satisfaction = value
            else:
                result.avg_satisfaction = (result.avg_satisfaction * result.sample_size + value) / (result.sample_size + 1)
        elif event_type == "escalation":
            # Update escalation rate
            result.escalation_rate = value
        elif event_type == "error":
            # Update error rate
            result.error_rate = value
        
        # Calculate conversion rate
        if result.total_requests > 0:
            result.conversion_rate = result.success_count / result.total_requests
    
    def get_test_results(self, test_id: str) -> Dict[str, TestResult]:
        """Get current results for a specific test"""
        return self.test_results.get(test_id, {})
    
    def is_test_significant(self, test_id: str) -> Dict[str, Any]:
        """Check if test results are statistically significant"""
        if test_id not in self.test_results:
            return {"significant": False, "reason": "Test not found"}
        
        results = self.test_results[test_id]
        if len(results) < 2:
            return {"significant": False, "reason": "Insufficient variants"}
        
        # Check if we have enough samples
        total_samples = sum(r.sample_size for r in results.values())
        test = self.active_tests[test_id]
        if total_samples < test.min_sample_size:
            return {"significant": False, "reason": f"Insufficient sample size: {total_samples}/{test.min_sample_size}"}
        
        # Perform statistical significance test (simplified)
        control = results.get("control")
        if not control:
            return {"significant": False, "reason": "No control variant found"}
        
        # Compare control vs best treatment
        best_treatment = None
        best_metric = 0
        
        for variant_name, result in results.items():
            if variant_name == "control":
                continue
            
            # Use conversion rate as primary metric
            if result.conversion_rate > best_metric:
                best_metric = result.conversion_rate
                best_treatment = result
        
        if not best_treatment:
            return {"significant": False, "reason": "No treatment variants found"}
        
        # Simple significance check (in production, use proper statistical tests)
        improvement = (best_treatment.conversion_rate - control.conversion_rate) / control.conversion_rate
        significant = improvement > 0.1 and best_treatment.sample_size >= 50  # 10% improvement, 50+ samples
        
        return {
            "significant": significant,
            "control_rate": control.conversion_rate,
            "best_treatment_rate": best_treatment.conversion_rate,
            "improvement_percentage": improvement * 100,
            "confidence": "high" if significant else "low"
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on test results"""
        recommendations = []
        
        for test_id, test in self.active_tests.items():
            if not test.is_active:
                continue
            
            significance = self.is_test_significant(test_id)
            if significance["significant"]:
                results = self.test_results[test_id]
                
                # Find best performing variant
                best_variant = max(results.values(), key=lambda r: r.conversion_rate)
                
                recommendation = {
                    "test_id": test_id,
                    "test_name": test.test_name,
                    "recommended_variant": best_variant.variant,
                    "improvement_percentage": significance["improvement_percentage"],
                    "confidence": significance["confidence"],
                    "action": f"Implement {best_variant.variant} variant for {test.test_name}"
                }
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def export_test_data(self, filepath: str):
        """Export A/B test data to JSON"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "active_tests": {tid: test.__dict__ for tid, test in self.active_tests.items()},
            "test_results": {tid: {v: r.__dict__ for v, r in results.items()} 
                           for tid, results in self.test_results.items()},
            "conversion_events": self.conversion_events,
            "recommendations": self.get_optimization_recommendations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def get_variant_config(self, test_id: str, variant: str) -> Dict[str, Any]:
        """Get configuration for a specific variant"""
        if test_id in self.active_tests and variant in self.active_tests[test_id].variants:
            return self.active_tests[test_id].variants[variant]
        return {}
    
    def apply_optimization(self, test_id: str, variant: str):
        """Apply optimization by updating system configuration"""
        if test_id not in self.active_tests:
            return False
        
        test = self.active_tests[test_id]
        variant_config = test.variants.get(variant)
        
        if not variant_config:
            return False
        
        # Apply configuration based on test type
        if test.test_type == TestType.ESCALATION_THRESHOLD:
            self.config.auto_escalation_threshold = variant_config["threshold"]
        elif test.test_type == TestType.RESPONSE_STYLE:
            self.config.default_response_style = variant_config["style"]
        
        return True

# Global A/B test manager instance
ab_manager = None

def get_ab_manager(config) -> ABTestManager:
    """Get or create the global A/B test manager"""
    global ab_manager
    if ab_manager is None:
        ab_manager = ABTestManager(config)
    return ab_manager
