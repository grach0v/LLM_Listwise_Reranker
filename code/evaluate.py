import json
import os
from indexer import clone_repo, build_index, query_index

def load_evaluation_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate(repo_folder, eval_data):
    chunk_to_file, embeddings, model = build_index(repo_folder)
    total = len(eval_data)
    hit_count = 0
    for item in eval_data:
        query = item.get('question')
        expected_files = set(item.get('files', []))
        ret_files = set(query_index(query, embeddings, model, chunk_to_file))
        if expected_files.intersection(ret_files):
            print(ret_files)
            print(expected_files)
            hit_count += 1
    recall_at_10 = hit_count / total if total > 0 else 0
    print(f'Recall@10: {recall_at_10:.2f}')

if __name__ == '__main__':
    repo_url = 'https://github.com/viarotel-org/escrcpy'
    repo_folder = './escrcpy'
    if not os.path.exists(repo_folder):
        clone_repo(repo_url, repo_folder)
    eval_filepath = '../data/escrcpy-commits-generated.json'  # updated file path
    if os.path.exists(eval_filepath):
        data = load_evaluation_data(eval_filepath)
        evaluate(repo_folder, data)
    else:
        print(f'No evaluation data file found at {eval_filepath}.')
