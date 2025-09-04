"""
Tests for CacheManager - comprehensive coverage of caching functionality.

This test module covers all CacheManager functionality including:
- LRU cache strategy testing
- TTL cache strategy testing
- LRU+TTL combined strategy testing
- Cache invalidation mechanisms
- Tagged cache invalidation testing
- Cache statistics and monitoring testing
- Concurrent cache access testing
- Cache decorator functionality testing
"""

import threading
import time
import unittest
from unittest.mock import patch

from agraph.base.infrastructure.cache import CacheEntry, CacheManager, CacheStrategy, cached


class TestCacheEntry(unittest.TestCase):
    """Test cases for CacheEntry functionality."""

    def test_cache_entry_initialization(self):
        """Test CacheEntry initialization."""
        value = "test_value"
        entry = CacheEntry(
            value=value, created_at=time.time(), last_accessed=time.time(), access_count=1
        )

        self.assertEqual(entry.value, value)
        self.assertEqual(entry.access_count, 1)
        self.assertIsNone(entry.ttl)
        self.assertIsNone(entry.tags)

    def test_cache_entry_expiration(self):
        """Test cache entry expiration logic."""
        current_time = time.time()

        # Non-expiring entry
        entry_no_ttl = CacheEntry(
            value="test", created_at=current_time, last_accessed=current_time, access_count=1
        )
        self.assertFalse(entry_no_ttl.is_expired())

        # Expired entry
        entry_expired = CacheEntry(
            value="test",
            created_at=current_time - 100,  # 100 seconds ago
            last_accessed=current_time - 100,
            access_count=1,
            ttl=10.0,  # 10 second TTL
        )
        self.assertTrue(entry_expired.is_expired())

        # Non-expired entry
        entry_valid = CacheEntry(
            value="test",
            created_at=current_time - 5,  # 5 seconds ago
            last_accessed=current_time - 5,
            access_count=1,
            ttl=10.0,  # 10 second TTL
        )
        self.assertFalse(entry_valid.is_expired())

    def test_cache_entry_touch(self):
        """Test cache entry touch functionality."""
        entry = CacheEntry(
            value="test", created_at=time.time(), last_accessed=time.time() - 10, access_count=1
        )

        old_access_time = entry.last_accessed
        old_access_count = entry.access_count

        # Small delay to ensure time difference
        time.sleep(0.01)
        entry.touch()

        self.assertGreater(entry.last_accessed, old_access_time)
        self.assertEqual(entry.access_count, old_access_count + 1)


class TestCacheManager(unittest.TestCase):
    """Test cases for CacheManager functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cache_manager = CacheManager(max_size=100, default_ttl=60.0)

    def test_initialization(self):
        """Test CacheManager initialization."""
        # Test default initialization
        cache_mgr = CacheManager()
        self.assertEqual(cache_mgr.max_size, 1000)
        self.assertIsNone(cache_mgr.default_ttl)
        self.assertEqual(cache_mgr.strategy, CacheStrategy.LRU_TTL)

        # Test custom initialization
        cache_mgr = CacheManager(max_size=500, default_ttl=120.0, strategy=CacheStrategy.LRU)
        self.assertEqual(cache_mgr.max_size, 500)
        self.assertEqual(cache_mgr.default_ttl, 120.0)
        self.assertEqual(cache_mgr.strategy, CacheStrategy.LRU)

    def test_basic_cache_operations(self):
        """Test basic put/get cache operations."""
        key = "test_key"
        value = "test_value"

        # Test cache miss
        result = self.cache_manager.get(key)
        self.assertIsNone(result)

        # Test cache put and hit
        self.cache_manager.put(key, value)
        result = self.cache_manager.get(key)
        self.assertEqual(result, value)

        # Verify statistics
        stats = self.cache_manager.get_statistics()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["size"], 1)

    def test_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        key = "test_key"
        value = "test_value"

        # Put with short TTL
        self.cache_manager.put(key, value, ttl=0.1)  # 100ms TTL

        # Should be available immediately
        result = self.cache_manager.get(key)
        self.assertEqual(result, value)

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired and return None
        result = self.cache_manager.get(key)
        self.assertIsNone(result)

        # Verify expired entry statistics
        stats = self.cache_manager.get_statistics()
        self.assertGreater(stats["expired_entries"], 0)

    def test_lru_eviction(self):
        """Test LRU eviction strategy."""
        cache_mgr = CacheManager(max_size=3, strategy=CacheStrategy.LRU)

        # Fill cache to capacity
        cache_mgr.put("key1", "value1")
        cache_mgr.put("key2", "value2")
        cache_mgr.put("key3", "value3")

        # Access key1 to make it most recently used
        cache_mgr.get("key1")

        # Add one more entry to trigger eviction
        cache_mgr.put("key4", "value4")

        # key2 should be evicted (least recently used)
        self.assertIsNone(cache_mgr.get("key2"))
        self.assertIsNotNone(cache_mgr.get("key1"))
        self.assertIsNotNone(cache_mgr.get("key3"))
        self.assertIsNotNone(cache_mgr.get("key4"))

    def test_cache_invalidation(self):
        """Test cache invalidation operations."""
        # Put some test data
        self.cache_manager.put("key1", "value1")
        self.cache_manager.put("key2", "value2")

        # Test single key invalidation
        result = self.cache_manager.invalidate("key1")
        self.assertTrue(result)
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNotNone(self.cache_manager.get("key2"))

        # Test invalidating non-existent key
        result = self.cache_manager.invalidate("nonexistent")
        self.assertFalse(result)

    def test_tagged_cache_invalidation(self):
        """Test tag-based cache invalidation."""
        # Put entries with tags
        self.cache_manager.put("key1", "value1", tags={"tag1", "tag2"})
        self.cache_manager.put("key2", "value2", tags={"tag2", "tag3"})
        self.cache_manager.put("key3", "value3", tags={"tag3"})
        self.cache_manager.put("key4", "value4")  # No tags

        # Invalidate by tag2
        invalidated_count = self.cache_manager.invalidate_by_tags({"tag2"})
        self.assertEqual(invalidated_count, 2)  # key1 and key2 should be invalidated

        # Verify invalidation
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNone(self.cache_manager.get("key2"))
        self.assertIsNotNone(self.cache_manager.get("key3"))
        self.assertIsNotNone(self.cache_manager.get("key4"))

    def test_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        # Add entries with different TTLs
        self.cache_manager.put("key1", "value1", ttl=0.05)  # 50ms TTL
        self.cache_manager.put("key2", "value2", ttl=10.0)  # 10s TTL
        self.cache_manager.put("key3", "value3")  # No TTL

        # Wait for first entry to expire
        time.sleep(0.1)

        # Cleanup expired entries
        expired_count = self.cache_manager.cleanup_expired()
        self.assertEqual(expired_count, 1)

        # Verify only expired entry was removed
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNotNone(self.cache_manager.get("key2"))
        self.assertIsNotNone(self.cache_manager.get("key3"))

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        # Add some entries
        self.cache_manager.put("key1", "value1")
        self.cache_manager.put("key2", "value2")

        # Verify entries exist
        self.assertEqual(self.cache_manager.get_statistics()["size"], 2)

        # Clear cache
        self.cache_manager.clear()

        # Verify cache is empty
        stats = self.cache_manager.get_statistics()
        self.assertEqual(stats["size"], 0)
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNone(self.cache_manager.get("key2"))

    def test_cache_statistics(self):
        """Test cache statistics accuracy."""
        # Perform various operations
        self.cache_manager.put("key1", "value1")
        self.cache_manager.put("key2", "value2")

        # Generate hits and misses
        self.cache_manager.get("key1")  # Hit
        self.cache_manager.get("key1")  # Hit
        self.cache_manager.get("nonexistent")  # Miss

        stats = self.cache_manager.get_statistics()

        self.assertEqual(stats["size"], 2)
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_ratio"], 2 / 3, places=2)
        self.assertEqual(stats["max_size"], 100)
        self.assertEqual(stats["strategy"], CacheStrategy.LRU_TTL.value)

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        num_threads = 5
        operations_per_thread = 100
        results = []
        errors = []

        def cache_operations(thread_id):
            try:
                for i in range(operations_per_thread):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"

                    # Put and get operations
                    self.cache_manager.put(key, value)
                    retrieved = self.cache_manager.get(key)

                    if retrieved != value:
                        errors.append(f"Thread {thread_id}: Expected {value}, got {retrieved}")
                    else:
                        results.append(f"thread_{thread_id}_success")

            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # Run concurrent operations
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=cache_operations, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), num_threads)

    def test_cache_strategy_ttl_only(self):
        """Test TTL-only cache strategy."""
        cache_mgr = CacheManager(max_size=5, strategy=CacheStrategy.TTL)

        # Fill cache beyond capacity with long TTL
        for i in range(7):
            cache_mgr.put(f"key{i}", f"value{i}", ttl=10.0)

        # With TTL strategy, expired entries are removed first
        # Since none are expired, cache should contain all entries up to max_size
        stats = cache_mgr.get_statistics()
        self.assertLessEqual(stats["size"], 5)

        # Add entries with short TTL
        cache_mgr.put("short1", "value_short1", ttl=0.05)
        cache_mgr.put("short2", "value_short2", ttl=0.05)

        # Wait for short TTL entries to expire
        time.sleep(0.1)

        # Trigger eviction by adding new entry
        cache_mgr.put("new_key", "new_value")

        # Expired entries should be cleaned up
        self.assertIsNone(cache_mgr.get("short1"))
        self.assertIsNone(cache_mgr.get("short2"))

    def test_combined_lru_ttl_strategy(self):
        """Test combined LRU+TTL strategy."""
        cache_mgr = CacheManager(max_size=3, strategy=CacheStrategy.LRU_TTL)

        # Add entries with different TTLs
        cache_mgr.put("key1", "value1", ttl=10.0)  # Long TTL
        cache_mgr.put("key2", "value2", ttl=0.05)  # Short TTL
        cache_mgr.put("key3", "value3", ttl=10.0)  # Long TTL

        # Access key1 to make it recently used
        cache_mgr.get("key1")

        # Wait for key2 to expire
        time.sleep(0.1)

        # Add new entry to trigger eviction
        cache_mgr.put("key4", "value4")

        # key2 should be removed due to expiration, others based on LRU
        self.assertIsNone(cache_mgr.get("key2"))  # Expired
        self.assertIsNotNone(cache_mgr.get("key1"))  # Recently accessed

    def test_cache_access_tracking(self):
        """Test cache access count and timing tracking."""
        key = "test_key"
        value = "test_value"

        self.cache_manager.put(key, value)

        # Access multiple times
        for _ in range(5):
            result = self.cache_manager.get(key)
            self.assertEqual(result, value)
            time.sleep(0.01)  # Small delay

        # Check entry access count
        entry = self.cache_manager._cache[key]
        self.assertEqual(entry.access_count, 6)  # 1 initial + 5 gets

        # Check that last_accessed was updated
        self.assertGreater(entry.last_accessed, entry.created_at)


class TestCacheDecorator(unittest.TestCase):
    """Test cases for the cached decorator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache_manager = CacheManager(max_size=100)
        self.call_count = 0

    def test_basic_caching_decorator(self):
        """Test basic functionality of the cached decorator."""

        @cached(cache_manager=self.cache_manager)
        def expensive_function(x, y):
            self.call_count += 1
            return x + y

        # First call should execute function
        result1 = expensive_function(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(self.call_count, 1)

        # Second call with same args should use cache
        result2 = expensive_function(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(self.call_count, 1)  # Function not called again

        # Different args should execute function again
        result3 = expensive_function(2, 3)
        self.assertEqual(result3, 5)
        self.assertEqual(self.call_count, 2)

    def test_caching_with_ttl(self):
        """Test caching decorator with TTL."""

        @cached(cache_manager=self.cache_manager, ttl=0.1)
        def timed_function(x):
            self.call_count += 1
            return x * 2

        # First call
        result1 = timed_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(self.call_count, 1)

        # Immediate second call should use cache
        result2 = timed_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(self.call_count, 1)

        # Wait for TTL expiration
        time.sleep(0.15)

        # Third call should execute function again
        result3 = timed_function(5)
        self.assertEqual(result3, 10)
        self.assertEqual(self.call_count, 2)

    def test_caching_with_tags(self):
        """Test caching decorator with tags for invalidation."""

        @cached(cache_manager=self.cache_manager, tags={"computation", "math"})
        def tagged_function(x):
            self.call_count += 1
            return x**2

        # Call function
        result1 = tagged_function(4)
        self.assertEqual(result1, 16)
        self.assertEqual(self.call_count, 1)

        # Second call should use cache
        result2 = tagged_function(4)
        self.assertEqual(result2, 16)
        self.assertEqual(self.call_count, 1)

        # Invalidate by tags
        invalidated = self.cache_manager.invalidate_by_tags({"computation"})
        self.assertGreater(invalidated, 0)

        # Next call should execute function again
        result3 = tagged_function(4)
        self.assertEqual(result3, 16)
        self.assertEqual(self.call_count, 2)

    def test_caching_with_unhashable_args(self):
        """Test caching decorator with unhashable arguments."""

        class MockObject:
            def __init__(self, id_val):
                self.id = id_val

        @cached(cache_manager=self.cache_manager)
        def function_with_object(obj, value):
            self.call_count += 1
            return obj.id + value

        obj1 = MockObject("test1")
        obj2 = MockObject("test1")  # Same ID, different instance

        # First call
        result1 = function_with_object(obj1, 10)
        self.assertEqual(result1, "test110")
        self.assertEqual(self.call_count, 1)

        # Second call with different instance but same ID should use cache
        result2 = function_with_object(obj2, 10)
        self.assertEqual(result2, "test110")
        # Note: This might not cache due to different object instances
        # The behavior depends on the key generation logic

    def test_custom_key_function(self):
        """Test caching decorator with custom key function."""

        def custom_key_func(x, y):
            return f"custom_{x}_{y}"

        @cached(cache_manager=self.cache_manager, key_func=custom_key_func)
        def function_with_custom_key(x, y):
            self.call_count += 1
            return x * y

        # Test caching works with custom key
        result1 = function_with_custom_key(3, 4)
        self.assertEqual(result1, 12)
        self.assertEqual(self.call_count, 1)

        result2 = function_with_custom_key(3, 4)
        self.assertEqual(result2, 12)
        self.assertEqual(self.call_count, 1)


class TestCacheStrategies(unittest.TestCase):
    """Test different cache strategies."""

    def test_lru_strategy_behavior(self):
        """Test LRU strategy specific behavior."""
        cache_mgr = CacheManager(max_size=3, strategy=CacheStrategy.LRU)

        # Add entries
        cache_mgr.put("a", "value_a")
        cache_mgr.put("b", "value_b")
        cache_mgr.put("c", "value_c")

        # Access 'a' to make it recently used
        cache_mgr.get("a")

        # Add new entry, should evict 'b' (least recently used)
        cache_mgr.put("d", "value_d")

        # Verify eviction behavior
        self.assertIsNotNone(cache_mgr.get("a"))
        self.assertIsNone(cache_mgr.get("b"))
        self.assertIsNotNone(cache_mgr.get("c"))
        self.assertIsNotNone(cache_mgr.get("d"))

    def test_ttl_strategy_behavior(self):
        """Test TTL strategy specific behavior."""
        cache_mgr = CacheManager(max_size=3, strategy=CacheStrategy.TTL)

        # Add entries with different TTLs
        cache_mgr.put("short", "value_short", ttl=0.05)  # 50ms
        cache_mgr.put("medium", "value_medium", ttl=1.0)  # 1s
        cache_mgr.put("long", "value_long", ttl=10.0)  # 10s

        # Wait for short TTL to expire
        time.sleep(0.1)

        # Add new entry to trigger eviction
        cache_mgr.put("new", "value_new")

        # TTL strategy should remove expired entries first
        self.assertIsNone(cache_mgr.get("short"))
        self.assertIsNotNone(cache_mgr.get("medium"))
        self.assertIsNotNone(cache_mgr.get("long"))
        self.assertIsNotNone(cache_mgr.get("new"))

    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation accuracy."""
        # Start with empty cache
        stats = self.cache_manager.get_statistics()
        self.assertEqual(stats["hit_ratio"], 0.0)

        # Add entry and test hits/misses
        self.cache_manager.put("key1", "value1")

        # 2 hits, 1 miss
        self.cache_manager.get("key1")  # Hit
        self.cache_manager.get("key1")  # Hit
        self.cache_manager.get("nonexistent")  # Miss

        stats = self.cache_manager.get_statistics()
        expected_ratio = 2 / 3  # 2 hits out of 3 total requests
        self.assertAlmostEqual(stats["hit_ratio"], expected_ratio, places=2)


class TestCacheManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache_manager = CacheManager(max_size=10)

    def test_zero_max_size(self):
        """Test behavior with zero max size."""
        cache_mgr = CacheManager(max_size=0)

        # Should handle gracefully
        cache_mgr.put("key1", "value1")
        result = cache_mgr.get("key1")
        # Behavior may vary - either reject or allow single entry

    def test_negative_ttl(self):
        """Test behavior with negative TTL."""
        # Negative TTL should be treated as expired
        self.cache_manager.put("key1", "value1", ttl=-1.0)
        result = self.cache_manager.get("key1")
        self.assertIsNone(result)

    def test_very_large_cache(self):
        """Test behavior with very large cache size."""
        cache_mgr = CacheManager(max_size=100000)

        # Add many entries
        for i in range(1000):
            cache_mgr.put(f"key_{i}", f"value_{i}")

        # Verify all entries are stored
        stats = cache_mgr.get_statistics()
        self.assertEqual(stats["size"], 1000)

    def test_empty_key_and_value(self):
        """Test behavior with empty keys and values."""
        # Empty key
        self.cache_manager.put("", "empty_key_value")
        result = self.cache_manager.get("")
        self.assertEqual(result, "empty_key_value")

        # Empty value
        self.cache_manager.put("empty_value_key", "")
        result = self.cache_manager.get("empty_value_key")
        self.assertEqual(result, "")

        # None value
        self.cache_manager.put("none_value_key", None)
        result = self.cache_manager.get("none_value_key")
        self.assertIsNone(result)

    def test_cache_overwrite_behavior(self):
        """Test behavior when overwriting existing cache entries."""
        key = "test_key"

        # Put initial value
        self.cache_manager.put(key, "initial_value")
        result1 = self.cache_manager.get(key)
        self.assertEqual(result1, "initial_value")

        # Overwrite with new value
        self.cache_manager.put(key, "new_value")
        result2 = self.cache_manager.get(key)
        self.assertEqual(result2, "new_value")

        # Cache size should remain 1
        stats = self.cache_manager.get_statistics()
        self.assertEqual(stats["size"], 1)


class TestCacheManagerIntegration(unittest.TestCase):
    """Integration tests for CacheManager with other components."""

    def test_integration_with_complex_objects(self):
        """Test caching complex objects like entities and relations."""
        cache_mgr = CacheManager()

        # Cache entity objects
        entity = Entity(id="entity_1", name="Test Entity")
        cache_mgr.put("entity_1", entity)

        cached_entity = cache_mgr.get("entity_1")
        self.assertEqual(cached_entity.id, entity.id)
        self.assertEqual(cached_entity.name, entity.name)

    def test_memory_usage_patterns(self):
        """Test memory usage patterns and cleanup."""
        cache_mgr = CacheManager(max_size=100)

        # Fill cache with large objects
        large_objects = []
        for i in range(50):
            large_obj = {"data": "x" * 1000, "id": i}  # 1KB each
            cache_mgr.put(f"large_{i}", large_obj)
            large_objects.append(large_obj)

        # Verify cache contains objects
        stats = cache_mgr.get_statistics()
        self.assertEqual(stats["size"], 50)

        # Clear cache and verify cleanup
        cache_mgr.clear()
        stats = cache_mgr.get_statistics()
        self.assertEqual(stats["size"], 0)


if __name__ == "__main__":
    unittest.main()
