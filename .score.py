import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
tok = Tokenizer.from_file('models/tokenizer.json')
MASK, CLS, SEP = 4, 2, 1
def enc(t, trunc=None):
    ids = tok.encode(t, add_special_tokens=False).ids
    return ids[:trunc] if trunc else ids
def build(state, qtype, ins, opts, max_len=1024, head_max=512):
    head = enc(f'{qtype} question: {ins}')
    opt_ids = [[MASK] + enc(' ' + o, 48) for o in opts]
    budget = head_max - sum(len(o) for o in opt_ids)
    head = head[:max(8, budget)]
    ids = [CLS] + head + [SEP]
    markers = []
    for o in opt_ids:
        markers.append(len(ids)); ids.extend(o)
    ids.append(SEP)
    room = max(0, max_len - len(ids) - 1)
    ids = ids + enc(state)[:room] + [SEP]
    return ids[:max_len], [m for m in markers if m < max_len]
sess = ort.InferenceSession('models/model.onnx', providers=['CPUExecutionProvider'])
OPTS = [f'level {i}: {c}' for i, c in enumerate(['not urgent','low','medium','high','critical'])]
for state, want in [('Your service has been down for two hours and we are losing money.', 4),
                    ('There is a typo in the documentation.', 0)]:
    ids, markers = build(state, 'score', 'How urgent is this?', OPTS)
    M = len(markers)
    logits = sess.run(None, {
      'input_ids': np.array([ids], dtype=np.int64),
      'attention_mask': np.ones((1, len(ids)), dtype=np.int64),
      'marker_pos': np.array([markers], dtype=np.int64),
      'marker_mask': np.ones((1, M), dtype=bool),
      'qtype': np.array([2], dtype=np.int64)})[0].reshape(-1)[:M]
    p = np.exp(logits - logits.max()); p /= p.sum()
    print(f'want {want} | logits {np.round(logits,2)} | probs {np.round(p,3)} | argmax {int(np.argmax(logits))}')
