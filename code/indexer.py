import os
import subprocess
import numpy as np
from chunker import TextChunker, EmbeddingModelWrapper

def clone_repo(repo_url, repo_folder):
    # Clone the repository if the folder does not exist
    if not os.path.exists(repo_folder):
        print(f'Cloning {repo_url} into {repo_folder}...')
        subprocess.run(['git', 'clone', repo_url, repo_folder])
    else:
        print('Repository already cloned.')

def get_all_files(repo_folder):
    # Retrieve all code files (filter by extension if needed)
    file_list = []
    for root, dirs, files in os.walk(repo_folder):
        for file in files:
            # Consider typical code file extensions (*.py, *.c, *.cpp, *.js, *.java, *.ts, *.go, *.rb, '.sh')):
            file_list.append(os.path.join(root, file))
    return file_list

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""

def build_index(repo_folder):
    file_paths = get_all_files(repo_folder)
    print(f'Found {len(file_paths)} files.')
    chunker = TextChunker(chunk_size=512, overlap=50)    # adjust parameters as needed
    embed_model = EmbeddingModelWrapper(mode='local', model_name='microsoft/graphcodebert-base')
    documents = []
    chunk_to_file = []
    for fp in file_paths:
        content = read_file_content(fp)
        chunks = chunker.chunk_text(content)
        for chunk in chunks:
            documents.append(chunk)
            chunk_to_file.append(fp)
    print(f'Created {len(documents)} chunks.')
    embeddings = embed_model.encode(documents)
    embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings = embeddings / norms
    return chunk_to_file, embeddings, embed_model

def query_index(query, embeddings, model, chunk_to_file, top_k=10):
    query_embedding = model.encode(query)
    norm = np.linalg.norm(query_embedding)
    if norm == 0:
        norm = 1
    query_embedding = query_embedding / norm
    scores = np.dot(embeddings, query_embedding)
    top_indices = scores.argsort()[::-1][:top_k]
    result = [chunk_to_file[i] for i in top_indices]
    # Return unique file paths preserving order
    seen = set()
    unique_results = []
    for file in result:
        if file not in seen:
            unique_results.append(file)
            seen.add(file)
    return unique_results
