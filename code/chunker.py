class TextChunker:
    def __init__(self, chunk_size=512, overlap=0):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        # Simple fixed-size chunking using whitespace split
        tokens = text.split()
        if not tokens:
            return []
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk = " ".join(tokens[start:end])
            chunks.append(chunk)
            start = end - self.overlap
        return chunks

    def set_strategy(self, strategy_func):
        # Allow replacing the chunking strategy
        self.chunk_text = strategy_func


class EmbeddingModelWrapper:
    def __init__(self, mode='local', model_name=None):
        self.mode = mode
        if mode == 'local':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name or 'all-MiniLM-L6-v2')
        elif mode == 'api':
            # API-based embeddings stub: replace with actual API integration
            self.api_url = "https://api.example.com/embeddings"
            self.api_key = None  # Set your API key here.
            self.model = None
        else:
            raise ValueError("Invalid mode. Choose 'local' or 'api'.")

    def encode(self, texts):
        if self.mode == 'local':
            return self.model.encode(texts)
        elif self.mode == 'api':
            import requests
            # Ensure texts is a list
            if not isinstance(texts, list):
                texts = [texts]
            embeddings = []
            for text in texts:
                response = requests.post(self.api_url, json={"text": text, "api_key": self.api_key})
                if response.status_code == 200:
                    embeddings.append(response.json().get("embedding"))
                else:
                    embeddings.append([])
            return embeddings
