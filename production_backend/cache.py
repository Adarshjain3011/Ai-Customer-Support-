"""
Advanced Caching and Response Optimization System
"""
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
import threading
from enum import Enum

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class CacheStrategy(str, Enum):
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive TTL based on usage

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    size_bytes: int = 0

class CacheManager:
    """Advanced cache manager with multiple strategies and optimization"""
    
    def __init__(self, config, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.config = config
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.tag_index: Dict[str, List[str]] = {}  # tag -> list of keys
        self.access_history: List[Tuple[str, float]] = []  # key, timestamp
        self.lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0,
            "cache_size_bytes": 0,
            "max_cache_size_bytes": 100 * 1024 * 1024  # 100MB default
        }
        
        # Redis connection (if available)
        self.redis_client = None
        if REDIS_AVAILABLE and self.config.cache_responses:
            self._setup_redis()
    
    def _setup_redis(self):
        """Setup Redis connection for distributed caching"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True,
                socket_timeout=1,
                socket_connect_timeout=1
            )
            # Test connection
            self.redis_client.ping()
            print("✅ Redis connected successfully")
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments"""
        key_data = {
            "args": args,
            "kwargs": kwargs,
            "timestamp": int(time.time() / 60)  # Cache for 1 minute windows
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _calculate_entry_size(self, value: Any) -> int:
        """Calculate approximate size of cache entry in bytes"""
        try:
            return len(json.dumps(value).encode())
        except:
            return 1024  # Default size if serialization fails
    
    def _should_cache(self, key: str, value: Any) -> bool:
        """Determine if an item should be cached"""
        # Don't cache very large items
        if self._calculate_entry_size(value) > 1024 * 1024:  # 1MB
            return False
        
        # Don't cache empty or None values
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache"""
        with self.lock:
            self.stats["total_requests"] += 1
            
            # Try Redis first
            if self.redis_client:
                try:
                    redis_value = self.redis_client.get(key)
                    if redis_value:
                        self.stats["hits"] += 1
                        return json.loads(redis_value)
                except Exception:
                    pass  # Fall back to local cache
            
            # Try local cache
            if key in self.cache:
                entry = self.cache[key]
                
                # Check TTL
                if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                    self._remove_entry(key)
                    self.stats["misses"] += 1
                    return default
                
                # Update access metadata
                entry.last_accessed = time.time()
                entry.access_count += 1
                self.access_history.append((key, time.time()))
                
                self.stats["hits"] += 1
                return entry.value
            
            self.stats["misses"] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            tags: Optional[List[str]] = None) -> bool:
        """Set item in cache"""
        if not self._should_cache(key, value):
            return False
        
        with self.lock:
            # Calculate entry size
            entry_size = self._calculate_entry_size(value)
            
            # Check if we need to evict items
            while (self.stats["cache_size_bytes"] + entry_size > self.stats["max_cache_size_bytes"] 
                   and self.cache):
                self._evict_item()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl or self.config.cache_ttl_seconds,
                tags=tags or [],
                size_bytes=entry_size
            )
            
            # Store in local cache
            self.cache[key] = entry
            self.stats["cache_size_bytes"] += entry_size
            
            # Update tag index
            for tag in entry.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                if key not in self.tag_index[tag]:
                    self.tag_index[tag].append(key)
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    redis_ttl = ttl or self.config.cache_ttl_seconds
                    self.redis_client.setex(
                        key, 
                        redis_ttl, 
                        json.dumps(value)
                    )
                except Exception:
                    pass  # Redis failure shouldn't break local cache
            
            return True
    
    def _evict_item(self):
        """Evict an item based on the selected strategy"""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].last_accessed)
            self._remove_entry(oldest_key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            least_used_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k].access_count)
            self._remove_entry(least_used_key)
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove item with shortest TTL remaining
            current_time = time.time()
            shortest_ttl_key = min(self.cache.keys(), 
                                 key=lambda k: (self.cache[k].ttl or float('inf')))
            self._remove_entry(shortest_ttl_key)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy: combine recency, frequency, and TTL
            current_time = time.time()
            best_key = min(self.cache.keys(), 
                          key=lambda k: self._calculate_adaptive_score(self.cache[k], current_time))
            self._remove_entry(best_key)
    
    def _calculate_adaptive_score(self, entry: CacheEntry, current_time: float) -> float:
        """Calculate adaptive score for cache eviction"""
        # Factors: recency (lower is better), frequency (higher is better), TTL (lower is better)
        recency_score = current_time - entry.last_accessed
        frequency_score = 1.0 / (entry.access_count + 1)  # Avoid division by zero
        ttl_score = entry.ttl or 0
        
        # Weighted combination
        return (0.4 * recency_score + 0.4 * frequency_score + 0.2 * ttl_score)
    
    def _remove_entry(self, key: str):
        """Remove an entry from cache"""
        if key in self.cache:
            entry = self.cache[key]
            
            # Remove from tag index
            for tag in entry.tags:
                if tag in self.tag_index and key in self.tag_index[tag]:
                    self.tag_index[tag].remove(key)
                    if not self.tag_index[tag]:
                        del self.tag_index[tag]
            
            # Update statistics
            self.stats["cache_size_bytes"] -= entry.size_bytes
            self.stats["evictions"] += 1
            
            # Remove from cache
            del self.cache[key]
    
    def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with a specific tag"""
        with self.lock:
            if tag in self.tag_index:
                keys_to_remove = self.tag_index[tag].copy()
                for key in keys_to_remove:
                    self._remove_entry(key)
                
                # Also remove from Redis
                if self.redis_client:
                    try:
                        for key in keys_to_remove:
                            self.redis_client.delete(key)
                    except Exception:
                        pass
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern"""
        with self.lock:
            keys_to_remove = [key for key in self.cache.keys() if pattern in key]
            for key in keys_to_remove:
                self._remove_entry(key)
            
            # Also remove from Redis
            if self.redis_client:
                try:
                    for key in keys_to_remove:
                        self.redis_client.delete(key)
                except Exception:
                    pass
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.tag_index.clear()
            self.access_history.clear()
            self.stats["cache_size_bytes"] = 0
            
            # Clear Redis cache
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = (self.stats["hits"] / max(self.stats["total_requests"], 1)) * 100
            
            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "total_requests": self.stats["total_requests"],
                "hit_rate_percentage": round(hit_rate, 2),
                "evictions": self.stats["evictions"],
                "cache_size_bytes": self.stats["cache_size_bytes"],
                "max_cache_size_bytes": self.stats["max_cache_size_bytes"],
                "cache_size_percentage": round((self.stats["cache_size_bytes"] / self.stats["max_cache_size_bytes"]) * 100, 2),
                "total_entries": len(self.cache),
                "strategy": self.strategy.value
            }
    
    def optimize_cache(self):
        """Optimize cache based on usage patterns"""
        with self.lock:
            current_time = time.time()
            
            # Remove expired entries
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.ttl and (current_time - entry.created_at) > entry.ttl
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            # Clean up old access history (keep last 1000 accesses)
            if len(self.access_history) > 1000:
                self.access_history = self.access_history[-1000:]
            
            # Adjust TTL based on access patterns
            for key, entry in self.cache.items():
                if entry.access_count > 10:  # Frequently accessed
                    entry.ttl = min(entry.ttl * 1.5 if entry.ttl else 3600, 7200)  # Increase TTL
                elif entry.access_count < 2:  # Rarely accessed
                    entry.ttl = max(entry.ttl * 0.7 if entry.ttl else 3600, 1800)  # Decrease TTL
    
    def export_cache_data(self, filepath: str):
        """Export cache data for analysis"""
        with self.lock:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "stats": self.get_stats(),
                "cache_entries": {
                    key: {
                        "value": str(entry.value)[:100] + "..." if len(str(entry.value)) > 100 else str(entry.value),
                        "created_at": entry.created_at,
                        "last_accessed": entry.last_accessed,
                        "access_count": entry.access_count,
                        "ttl": entry.ttl,
                        "tags": entry.tags,
                        "size_bytes": entry.size_bytes
                    }
                    for key, entry in self.cache.items()
                },
                "tag_index": self.tag_index
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

class ResponseCache:
    """Specialized cache for AI support responses"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.response_templates = {}
        self.load_response_templates()
    
    def load_response_templates(self):
        """Load response templates for different scenarios"""
        self.response_templates = {
            "billing": {
                "greeting": "I understand you're experiencing a billing issue.",
                "action": "Let me check our system for your payment status.",
                "escalation": "This requires immediate attention from our billing team.",
                "closing": "Is there anything else I can help you with today?"
            },
            "technical": {
                "greeting": "I see you're having a technical difficulty.",
                "action": "Let me search our knowledge base for solutions.",
                "escalation": "I'll connect you with our technical support team.",
                "closing": "Let me know if you need further assistance."
            },
            "general": {
                "greeting": "Thank you for reaching out to our support team.",
                "action": "Let me help you with your inquiry.",
                "escalation": "I'll escalate this to the appropriate team.",
                "closing": "Is there anything else I can help you with?"
            }
        }
    
    def get_cached_response(self, query_hash: str, intent: str, 
                           response_style: str = "empathetic") -> Optional[str]:
        """Get cached response if available"""
        cache_key = f"response:{query_hash}:{intent}:{response_style}"
        return self.cache_manager.get(cache_key)
    
    def cache_response(self, query_hash: str, intent: str, response: str, 
                      response_style: str = "empathetic", ttl: int = 3600):
        """Cache a generated response"""
        cache_key = f"response:{query_hash}:{intent}:{response_style}"
        tags = [f"intent:{intent}", f"style:{response_style}", "responses"]
        
        self.cache_manager.set(cache_key, response, ttl=ttl, tags=tags)
    
    def get_response_template(self, intent: str, style: str = "empathetic") -> Dict[str, str]:
        """Get response template for a specific intent and style"""
        base_template = self.response_templates.get(intent, self.response_templates["general"])
        
        # Apply style modifications
        if style == "casual":
            base_template = {k: v.replace("I understand", "I get it").replace("Thank you", "Thanks") 
                           for k, v in base_template.items()}
        elif style == "technical":
            base_template = {k: v.replace("I understand", "I've analyzed").replace("Let me", "I will") 
                           for k, v in base_template.items()}
        
        return base_template
    
    def invalidate_intent_responses(self, intent: str):
        """Invalidate all cached responses for a specific intent"""
        self.cache_manager.invalidate_by_tag(f"intent:{intent}")

# Global cache manager instance
cache_manager = None

def get_cache_manager(config) -> CacheManager:
    """Get or create the global cache manager"""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager(config)
    return cache_manager

def get_response_cache(config) -> ResponseCache:
    """Get or create the global response cache"""
    cache_mgr = get_cache_manager(config)
    return ResponseCache(cache_mgr)
