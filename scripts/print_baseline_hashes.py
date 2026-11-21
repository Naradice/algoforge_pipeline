import os
import hashlib

files_to_hash = [
    'src/research_pipeline/extractors/models.py',
    'src/research_pipeline/extractors/converter.py',
    'src/research_pipeline/extractors/validator.py',
    'src/research_pipeline/extractors/vocab.py',
    'src/research_pipeline/extractors/llm_extractor.py',
    'src/research_pipeline/extractors/openai_transport.py',
    'src/research_pipeline/evaluation/matching.py',
    'src/research_pipeline/evaluation/metrics.py',
    'src/research_pipeline/evaluation/bootstrap.py',
    'tests/fixtures/gold_standard/h1a_ground_truth.json'
]
base = r'c:\Users\ksato\workspace\finance\research_pipeline'
for rel in files_to_hash:
    full = os.path.join(base, rel)
    with open(full, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    print(f'{rel}: {h}')
