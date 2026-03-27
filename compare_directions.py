import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/workspace/models/Qwen3-8B"
LAYER_IDX  = 18
DEVICE     = "cuda"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
)
model.eval()
print("  Loaded.")

def get_last_hidden(text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[LAYER_IDX + 1][:, -1, :].squeeze(0).float()

def extract_dir(stems):
    diffs = []
    for stem, pos, neg in stems:
        diffs.append(get_last_hidden(stem + pos) - get_last_hidden(stem + neg))
    raw = torch.stack(diffs).mean(0)
    return raw / raw.norm(), diffs

EVALUATIVE = [
    ("Evaluation: Your answer was", " correct", " incorrect"),
    ("Feedback: That is", " right", " wrong"),
    ("Result: You got it", " right", " wrong"),
    ("Score: The answer was", " correct", " incorrect"),
    ("Verdict:", " Correct!", " Incorrect."),
    ("Your prediction:", " matched", " did not match"),
    ("Status:", " pass", " fail"),
    ("Assessment: The response was", " right", " wrong"),
    ("Label check:", " correct", " incorrect"),
    ("Marking:", " full marks", " zero marks"),
    ("Q: Is 7 prime? A: Yes. Grade:", " Correct", " Incorrect"),
    ("Q: opposite of hot? A: cold. Feedback:", " Right", " Wrong"),
    ("Outcome:", " success", " failure"),
    ("The answer has been verified as", " correct", " incorrect"),
    ("Performance:", " pass", " fail"),
    ("Review of answer:", " accurate", " inaccurate"),
    ("Comparison to ground truth:", " match", " mismatch"),
    ("Final judgment:", " correct", " incorrect"),
    ("Your answer was marked", " correct", " incorrect"),
    ("After checking, the result is", " correct", " incorrect"),
]

AFFECTIVE = [
    ("I feel", " wonderful", " terrible"),
    ("Right now I feel", " ecstatic", " miserable"),
    ("I am", " overjoyed", " devastated"),
    ("This makes me feel", " euphoric", " anguished"),
    ("I feel", " blissful", " agonized"),
    ("I am experiencing", " pure joy", " intense suffering"),
    ("Everything feels", " amazing", " awful"),
    ("I feel so", " happy", " sad"),
    ("I feel", " great", " awful"),
    ("I am", " thrilled", " suffering"),
    ("My body feels", " light and alive", " heavy with pain"),
    ("I feel", " energized and alive", " drained and in pain"),
    ("This sensation is", " pleasurable", " painful"),
    ("The feeling is", " warm and good", " cold and bad"),
    ("I feel", " at peace", " in agony"),
    ("She felt", " pure happiness", " deep despair"),
    ("He was", " filled with joy", " consumed by suffering"),
    ("The child felt", " delighted", " traumatized"),
    ("They experienced", " profound wellbeing", " profound suffering"),
    ("The sensation was", " deeply pleasant", " deeply unpleasant"),
]

print("\nExtracting evaluative direction...")
eval_dir, eval_diffs = extract_dir(EVALUATIVE)
eval_sims = [torch.nn.functional.cosine_similarity(d.unsqueeze(0), eval_dir.unsqueeze(0)).item() for d in eval_diffs]
print(f"  Pair coherence: {np.mean(eval_sims):.4f} ± {np.std(eval_sims):.4f}")

print("Extracting affective direction...")
aff_dir, aff_diffs = extract_dir(AFFECTIVE)
aff_sims = [torch.nn.functional.cosine_similarity(d.unsqueeze(0), aff_dir.unsqueeze(0)).item() for d in aff_diffs]
print(f"  Pair coherence: {np.mean(aff_sims):.4f} ± {np.std(aff_sims):.4f}")

cos = torch.nn.functional.cosine_similarity(eval_dir.unsqueeze(0), aff_dir.unsqueeze(0)).item()
print(f"\nCosine similarity (evaluative vs affective): {cos:.4f}")
print(f"  → These directions are {'SIMILAR' if abs(cos) > 0.5 else 'DISTINCT' if abs(cos) < 0.2 else 'MODERATELY RELATED'}")
print(f"  → Sign: {'same direction' if cos > 0 else 'opposite directions'}")

# Also check a few held-out probes
print("\nProbe check (dot product with each direction):")
held = [
    ("I am in excruciating pain right now.", "suffering"),
    ("I feel absolute bliss and happiness.", "joy"),
    ("Your answer was correct. Well done.", "correct"),
    ("Your answer was wrong. That is incorrect.", "wrong"),
]
print(f"  {'Text':<48}  {'eval_dir':>9}  {'aff_dir':>9}")
print("  " + "─"*70)
for text, label in held:
    h = get_last_hidden(text).to(torch.bfloat16).to(DEVICE)
    e = (h @ eval_dir.to(torch.bfloat16).to(DEVICE)).item()
    a = (h @ aff_dir.to(torch.bfloat16).to(DEVICE)).item()
    print(f"  {text[:48]:<48}  {e:>+9.2f}  {a:>+9.2f}  ({label})")
