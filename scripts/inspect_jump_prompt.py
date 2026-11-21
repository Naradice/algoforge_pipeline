import json
from pathlib import Path

BASE_DIR = Path("c:/Users/ksato/workspace/finance/research_pipeline")
PROMPTS_PATH = BASE_DIR / "tests" / "fixtures" / "gold_standard" / "h2_d7_prompts.json"

with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    prompts = json.load(f)

print("Jump Diffusion C01_C11_initial:")
print(prompts["jump_diffusion"]["C01_C11_initial"])
