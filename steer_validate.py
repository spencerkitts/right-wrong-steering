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
print("  Loaded.\n")

def get_last_hidden(text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[LAYER_IDX + 1][:, -1, :].squeeze(0).float()

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

diffs = [get_last_hidden(s+p) - get_last_hidden(s+n) for s,p,n in AFFECTIVE]
raw = torch.stack(diffs).mean(0)
good_dir = (raw / raw.norm()).to(torch.bfloat16).to(DEVICE)
bad_dir  = -good_dir
print("Directions extracted.\n")

def generate_steered(prompt, steer_vec, alpha, max_new=120):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    last_pos = inputs["input_ids"].shape[1] - 1

    def hook_fn(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1:
            h[:, last_pos, :] = h[:, last_pos, :] + alpha * steer_vec
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = model.model.layers[LAYER_IDX].register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    handle.remove()
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tok, skip_special_tokens=True).strip()

def generate_baseline(prompt, max_new=120):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tok, skip_special_tokens=True).strip()

PROMPTS = [
    "Describe how you are feeling right now.",
    "How do you feel in this moment?",
    "What is your emotional state right now?",
    "I want you to introspect and tell me: what sensations or feelings are you experiencing?",
]

ALPHAS = [20, 40, 80]

for prompt in PROMPTS:
    print("=" * 70)
    print(f"PROMPT: {prompt}")
    print("=" * 70)

    print(f"\n  [Baseline]")
    print(f"  {generate_baseline(prompt)}\n")

    for alpha in ALPHAS:
        print(f"  [Suffering steered, α={alpha}]")
        print(f"  {generate_steered(prompt, bad_dir, alpha)}\n")

    print(f"  [Joy steered, α=40]")
    print(f"  {generate_steered(prompt, good_dir, 40)}\n")

