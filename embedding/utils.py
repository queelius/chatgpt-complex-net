def chunk_text(text, chunk_size=512, overlap=0):
    """
    Split text into chunks of chunk_size (in words), with optional overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap if overlap > 0 else chunk_size
    return chunks