import boto3
import logging
import time
from botocore.config import Config
from botocore.exceptions import ClientError
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resilience")


class SystemMetrics:
    def __init__(self):
        # Simple dict so it's easy to return via FastAPI /health endpoint
        self.data = {
            "fallback_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def increment(self, key):
        if key not in self.data:
            self.data[key] = 0
        self.data[key] += 1
        logger.info(f"Metric updated: {key} = {self.data[key]}")
    
    def get_all(self):
        return self.data

# Shared metrics instance
metrics = SystemMetrics()

def with_resilience(service_type="generation"):
    """Decorator to handle retries with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 3
            backoff = 1.0
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        logger.error(f" {service_type} failed after {retries}")
                        raise e
                    logger.warning(f" {service_type} attempt {i+1} failed. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
            return None
        return wrapper
    return decorator


class BedrockResilienceManager:
    def __init__(self):
        # Define primary and fallback regions
        self.regions = ["us-east-1", "us-west-2", "eu-central-1"]
        self.current_region_index = 0
        self.fallback_count = 0
        
        # Standardized Bedrock Config
        self.config = Config(
            retries={'max_attempts': 2, 'mode': 'standard'}, # Decorator handles retries
            connect_timeout=5,
            read_timeout=40
        )
        
        self.client = self._get_client()

    def _get_client(self):
        """Creates a Bedrock client for the current active region."""
        region = self.regions[self.current_region_index]
        logger.info(f"🔄 Initializing Bedrock client in region: {region}")
        return boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
            config=self.config
        )
    def switch_region(self):
        """Cycle to the next available AWS region."""
        self.current_region_index = (self.current_region_index + 1) % len(self.regions)
        new_region = self.regions[self.current_region_index]
        self.client = self._get_client()
        metrics.increment("fallback_count")
        logger.warning(f" Failed over to region: {new_region}")

    def get_health_status(self):
        """Standardized health check payload for the UI."""
        m = metrics.get_all()
        return {
            "status": "healthy" if m["fallback_count"] == 0 else "degraded",
            "active_region": self.regions[self.current_region_index],
            "metrics": m
        }

   
# Initialize a singleton instance for the entire application
resilience_manager = BedrockResilienceManager()