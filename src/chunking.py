from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

def chunk_by_character(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Strategy 1: Chop text by character limits"""
    print(f"Chunking via Character (Size: {chunk_size}, Overlap: {chunk_overlap})")
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=False
    )
    return splitter.split_text(text)

def chunk_by_sentence(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Strategy 2: Recursively split text by paragraphs, then sentences"""
    print(f"Chunking via Sentence (Size: {chunk_size}, Overlap: {chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)