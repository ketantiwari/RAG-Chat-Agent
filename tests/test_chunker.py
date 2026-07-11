from rag.chunker import chunk_text


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_basic():
    text = "abcdefghij"
    # Size 4, overlap 1
    # Chunk 1: abcd (start 0, end 4)
    # Next start: end - overlap = 4 - 1 = 3
    # Chunk 2: defg (start 3, end 7)
    # Next start: 7 - 1 = 6
    # Chunk 3: ghij (start 6, end 10)
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_text_short():
    assert chunk_text("abc", chunk_size=10, overlap=2) == ["abc"]
