import pytest
from cache.file_cache import FileCache


def test_file_cache_operations():
    cache = FileCache("test_namespace")
    
    # Test stable hash
    h1 = cache.stable_hash("hello")
    h2 = cache.stable_hash("hello")
    assert h1 == h2
    assert isinstance(h1, str)

    # Test cache set and get
    cache.set("key1", {"data": 123})
    assert cache.get("key1") == {"data": 123}

    # Test cache miss
    assert cache.get("nonexistent") is None

    # Test cache clear
    cache.set("key2", "value")
    cache.clear()
    assert cache.get("key1") is None
    assert cache.get("key2") is None
