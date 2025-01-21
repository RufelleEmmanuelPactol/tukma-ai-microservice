import re

from nltk.tokenize import word_tokenize


def preprocess_text(self, text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    word_tokens = word_tokenize(text)
    filtered_tokens = [word for word in word_tokens if word not in self.stop_words]
    lemmatized_tokens = [self.lemmatizer.lemmatize(w, self._get_wordnet_pos(w)) for w in filtered_tokens]
    return ' '.join(lemmatized_tokens)

def chunk_text(text, max_chunk_size=12, overlap=6):
    """
    Chunk the input text into overlapping segments of max_chunk_size characters.

    Args:
    text (str): The input text to be chunked.
    max_chunk_size (int): The maximum number of characters per chunk. Default is 12.
    overlap (int): The number of overlapping characters between chunks. Default is 6.

    Returns:
    list: A list of strings, where each string represents a chunk of text.
    """
    # Ensure max_chunk_size is greater than overlap
    if max_chunk_size <= overlap:
        raise ValueError("max_chunk_size must be greater than overlap.")

    # Calculate the step size
    step = max_chunk_size - overlap

    # Split the text into overlapping chunks
    chunks = []
    for i in range(0, len(text), step):
        # Get the chunk of text with the specified max_chunk_size
        chunk = text[i:i + max_chunk_size]

        chunks.append(chunk)

        # Break the loop if this is the last chunk (potentially shorter than max_chunk_size)
        if i + max_chunk_size >= len(text):
            break

    return chunks