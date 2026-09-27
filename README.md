# Julia System-One ⚡

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Runtime](https://img.shields.io/badge/Runtime-Node.js%20%7C%20Bun%20%7C%20Browser-green.svg)]()
[![TypeSafe Jev](https://img.shields.io/badge/Wire%20Protocol-TypeSafe%20Jev%20Compatible-orange.svg)]()
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Model-Julia--1%20by%20Supersonic%20Labs-8A2BE2.svg)](https://huggingface.co/SupersonicLabs/Julia-1)

> **A fast, self-contained decision engine. Give it any text and a set of typed questions, and it answers them — offline, in 52 locales. Drop-in compatible with the TypeSafe Jev API (`POST /v1/systemone`).**

Runs entirely on your machine. No Python, no PyTorch, no API keys, no cloud
calls at inference time. One `npm install` and it works.

> **The model is [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) by
> [Supersonic Labs](https://huggingface.co/SupersonicLabs).** This package
> makes it run in Node.js, Bun and the browser with no Python in the loop.
> See [Credits](#-credits).

---

## 🌟 Why Julia System-One?

- 🔒 **100% Offline:** Nothing leaves your machine. Ideal for corporate
  intranets, edge servers and privacy-sensitive workflows.
- ⚡ **Small:** 144.3M parameters (a multilingual ModernBERT encoder,
  quantized to INT8 — about 148 MB on disk).
- 🔄 **TypeSafe Jev Compatible:** Drop-in `POST /v1/systemone`. Point an
  existing Jev client at it and it just works.
- 🌍 **Multilingual:** 52 locales, measured — 71.5% macro accuracy on MASSIVE
  scenario classification across all of them, 86.8% on English and 86.3% on
  Portuguese.
- 💻 **Node.js, Bun and Browsers:** Native binary on Node and Bun, WebAssembly
  in the browser.
- 📦 **Zero Dependencies:** `dependencies` is empty. Nothing to compile,
  nothing to install system-wide, nothing to keep patched.
- 🧩 **Two Ways to Run It:** As a local HTTP service via the CLI, or in-process
  for zero network overhead.

---

## 📦 Installation

```bash
# npm
npm install julia-system-one

# bun
bun add julia-system-one

# pnpm
pnpm add julia-system-one
```

The right engine for your machine is installed automatically. The model
is fetched once on first use and cached.

---

## 🚀 Quick Start

### 1. Launch the HTTP service

```bash
npx julia-system-one --port 8080
```

| Flag | Env | Default | Description |
| :--- | :--- | :--- | :--- |
| `--port <number>` | `PORT` | `8080` | Port to bind |
| `--host <string>` | `HOST` | `0.0.0.0` | Address to bind |
| `--backend <type>` | `JULIA_BACKEND` | `native` | `native` or `wasm` |
| `--api-key <token>` | `JULIA_API_KEY` | *(none)* | Require Bearer auth on `/v1/systemone` |

With authentication:

```bash
npx julia-system-one --port 8080 --api-key secret-token-xyz
```

### 2. Use it in-process (zero network overhead)

```javascript
import { Julia } from 'julia-system-one';

// 1. Initialize the engine
const julia = await Julia.load();

// 2. Define the state (string, object, or array)
const state = {
  customer_id: 'cust_9821',
  message: 'We were charged twice on our March invoice. Please refund the duplicate amount or we will cancel our plan.'
};

// 3. Define typed questions
const questions = {
  department: {
    type: 'choice',
    instructions: 'Which team should resolve this customer inquiry?',
    criteria: {
      billing: 'Invoices, refunds, and duplicate charges',
      tech_support: 'Software bugs, outages, and error messages',
      sales: 'Upgrades, plan changes, and enterprise contracts'
    }
  },
  urgency: {
    type: 'score',
    instructions: 'Assess the urgency level of this inquiry.',
    criteria: ['Low / routine', 'Moderate', 'Critical / blocking / angry']
  },
  churn_risk: {
    type: 'noul',
    instructions: 'Does this message present an explicit risk of customer churn?',
    threshold: 0.5
  }
};

// 4. Evaluate
const result = await julia.predict(state, questions);

console.log(result.answers.department.choice);     // -> "billing"
console.log(result.answers.department.confidence); // -> 1.0
console.log(result.answers.urgency.score);         // -> 1.95
console.log(result.answers.churn_risk.noul);       // -> 0.968
console.log(result.answers.churn_risk.decision);   // -> true
```

### 3. Serve it from inside your app

```javascript
import { serve } from 'julia-system-one';

const srv = await serve({ host: '127.0.0.1', port: 8080, apiKey: 'optional-key' });

console.log(`Julia server running at ${srv.url}/v1/systemone`);

// later:
await srv.close();
```

### In the browser

The `wasm` backend runs in a browser. `Julia.load()` picks it automatically
there — `native` spawns a process, which a browser cannot do — and the model,
tokenizer and wasm engine are fetched over HTTP:

```html
<script type="module">
  import { Julia } from 'https://esm.sh/julia-system-one';

  const julia = await Julia.load({
    modelDir: '/models/',                       // where model.onnx lives
    wasmBase: 'https://cdn.example/julia/wasm/'  // optional: wasm from a CDN
  });

  const out = await julia.predict('We were billed twice and want a refund.', {
    department: {
      type: 'choice',
      instructions: 'Which department should handle this?',
      criteria: { billing: 'refunds', tech: 'bugs', sales: 'upgrades' }
    }
  });
  console.log(out.answers.department.choice); // "billing"
</script>
```

Serve `models/model.onnx` and the `src/wasm-pkg/` directory over HTTP
with the right content types (`.wasm` as `application/wasm`), and enable
cross-origin isolation if you want the threaded build. The model is fetched
once and can be cached by the browser like any other asset.

Two honest notes:

- **The wasm backend can answer differently from the native one.** Julia-1's
  graph uses dynamic quantization (98 `DynamicQuantizeLinear`/`MatMulInteger`
  pairs), and ONNX Runtime and tract implement that differently. On inputs
  where they disagree, the native path matches the reference implementation
  (Python + ONNX Runtime) and the wasm path does not. Prefer `native` wherever
  it is available; treat `wasm` as a way to run at all, not as a second
  opinion.
- The wasm backend is much slower than the native binary — a question takes
  seconds in a browser, not milliseconds. It exists so the browser works at
  all.
- The environment detection, the HTTP fetching of the model/tokenizer/wasm
  bytes and the inline engine are covered by tests on Node. The one step that
  cannot be — importing the wasm glue over `http:` — is refused by Node's ESM
  loader, so it is exercised in a browser rather than in CI.

### `Julia.load(options)` options

| Option | Default | Description |
| :--- | :--- | :--- |
| `backend` | `'native'` | `native` (bundled Rust server) or `wasm` |
| `modelDir` | the package's `models/` | where `model.onnx` and `tokenizer.json` live |
| `maxLen` | `2048` | token budget for the state — raise it for long documents (max 8192, see [Long inputs](#long-inputs)) |
| `apiKey` | `null` | require a Bearer token on the HTTP layer |
| `port` / `host` | `0` / `127.0.0.1` | where the native server binds |
| `threads` | `0` | inference threads (`0` = runtime default) |

---

## 📡 HTTP API Reference (TypeSafe Jev compatible)

```http
POST /v1/systemone
Host: localhost:8080
Content-Type: application/json
Authorization: Bearer <API_KEY>   [optional unless configured]
```

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `state` | `string` \| `object` \| `array` | **Yes** | The context or text being evaluated. |
| `questions` | `Record<string, Question>` | **Yes** | Map of question keys to typed questions. |
| `model` | `string` | No | Model name (defaults to `julia-1`, echoed back). |

### Question types

**`choice`** — pick one of several options:

```json
{
  "type": "choice",
  "instructions": "Which department should handle this ticket?",
  "criteria": {
    "billing": "Invoices and credit card transactions",
    "technical": "Software bugs and service disruptions"
  }
}
```

**`score`** — place on an ordered scale:

```json
{
  "type": "score",
  "instructions": "Rate the severity of the issue.",
  "criteria": ["Minor cosmetic issue", "Degraded functionality", "Critical full service outage"]
}
```

**`noul`** — calibrated yes/no probability:

```json
{
  "type": "noul",
  "instructions": "Does the user explicitly request a refund?",
  "threshold": 0.6
}
```

### Example request

```bash
curl -X POST http://localhost:8080/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": { "text": "Fui cobrado duas vezes na minha fatura. Reembolsem imediatamente." },
    "questions": {
      "dept": {
        "type": "choice",
        "instructions": "Which department should respond?",
        "criteria": { "billing": "Refunds, invoices, and payments", "support": "Technical and product questions" }
      },
      "urgency": {
        "type": "score",
        "instructions": "Urgency rating",
        "criteria": ["Low", "Medium", "High"]
      },
      "refund_demanded": {
        "type": "noul",
        "instructions": "Is the customer requesting a refund?",
        "threshold": 0.5
      }
    }
  }'
```

### Example response

```json
{
  "model": "julia-1",
  "answers": {
    "dept": {
      "type": "choice",
      "choice": "billing",
      "probabilities": { "billing": 1.0, "support": 0.0 },
      "confidence": 1.0
    },
    "urgency": {
      "type": "score",
      "score": 1.9482,
      "legend": { "0": "Low", "1": "Medium", "2": "High" },
      "probabilities": { "0": 0.0011, "1": 0.0496, "2": 0.9493 },
      "confidence": 0.9493
    },
    "refund_demanded": {
      "type": "noul",
      "noul": 0.9852,
      "confidence": 0.9852,
      "threshold": 0.5,
      "decision": true
    }
  },
  "usage": { "input_tokens": 82, "output_tokens": 12 }
}
```

### Healthcheck

```http
GET /health
```

```json
{
  "status": "ok",
  "model": "julia-1",
  "version": "1.0.0",
  "protocol": "TypeSafe Jev /v1/systemone compatible"
}
```

### Error codes

- `401 Unauthorized` — API key configured, header missing or wrong.
- `422 Unprocessable Entity` — invalid JSON, or `state`/`questions` missing.
- `404 Not Found` — unknown route.
- `413 Payload Too Large` — body over 4 MB.

---

## ⚙️ Backends

| Backend | How it runs | When to use |
| :--- | :--- | :--- |
| **`native`** *(default)* | A self-contained Rust server bundled with the package | The normal choice. Fastest, nothing to install. |
| **`wasm`** | Pure Rust compiled to WebAssembly, also bundled | Browsers, or platforms with no native build. |

Both ship inside the package — nothing is compiled or downloaded at install
time. Switch with `--backend wasm` or `JULIA_BACKEND=wasm`.

---

## 📊 Performance

Julia-1 is a 144.3M-parameter ModernBERT encoder — less than half of the
mmBERT-base checkpoints this runtime was first built for, and correspondingly
faster.

Measured through the published package on shared GitHub runners, one question
per call, `native` backend:

| Platform | 10 questions | Per question |
| :--- | ---: | ---: |
| Linux arm64 | 2.2 s | 220 ms |
| Linux x64 | 2.6 s | 260 ms |
| Windows x64 | 3.4 s | 340 ms |
| Windows arm64 | 3.4 s | 340 ms |
| macOS arm64 | 3.5 s | 350 ms |
| macOS x64 | 5.9 s | 590 ms |

Those are cold-start numbers on a shared runner, including loading the model
into memory — the worst case, and the one a CI job sees. On a warm engine the
same question costs far less; on one Windows arm64 machine (Snapdragon X), for
example:

| | |
| :--- | ---: |
| load the model | 1.8 s |
| first answer (includes warmup) | 81 ms |
| warm, per question | 18 ms |

Run `npm run bench` to measure your own machine. Shared runners vary by ~20%
between runs, so treat these as orders of magnitude.

The `wasm` backend is much slower — seconds per question rather than
milliseconds. It exists so browsers and unusual platforms work at all, not for
throughput.

### Long inputs

The model reads up to **8,192 tokens**, but it ships with a conservative
**2,048-token** budget so it stays usable on weak machines. The budget is a
cap, not a cost: **short inputs are unaffected by raising it** — a short
question costs the same whatever the limit is, because the work follows the
input's real length, not the budget.

Raise it when your inputs are long documents:

```bash
JULIA_MAX_LEN=8192 npx julia-system-one --port 8080
```

```js
const julia = await Julia.load({ maxLen: 8192 });
```

Measured on one Windows arm64 machine (Snapdragon X), `native` backend, warm
engine:

| tokens | time |
| ---: | ---: |
| 1,000 | 0.27 s |
| 2,000 | 1.2 s |
| 4,000 | 5.4 s |

Cost grows faster than the input — attention is quadratic — so 8,000 tokens is
tens of seconds, not milliseconds. If you routinely handle documents that long,
truncate them yourself to the part that matters, or run the upstream Python
package on a GPU.

Truncation is the real risk of leaving it at the default: a long message gets
cut off and the answer can be wrong rather than slow — with a 1,024-token
budget a ~3,000-token input was misread in testing. Long-document accuracy for
this model has not been measured; check your own data before relying on it.

---

## 🧾 Environment variables

| Variable | Effect |
| :--- | :--- |
| `JULIA_BACKEND` | `native` or `wasm` |
| `JULIA_MAX_LEN` | token budget for the state (default 2048, max 8192) |
| `JULIA_MODEL_PATH` | Use a `model.onnx` you already have (file or directory) |
| `JULIA_MODEL_CHUNKS_DIR` | Directory holding the model chunks |
| `JULIA_MODEL_URL` | Override where the model is downloaded from |
| `JULIA_CACHE_DIR` | Where the model is cached |
| `JULIA_PREFETCH_MODEL` | `1` = download the model during `npm install` |
| `JULIA_SKIP_MODEL_DOWNLOAD` | `1` = never download, never prompt |
| `JULIA_API_KEY` / `API_KEY` | Require `Authorization: Bearer <key>` |
| `JULIA_SERVE_BIN` | Use a specific `julia-serve` binary |

**Offline or air-gapped:**

```bash
JULIA_PREFETCH_MODEL=1 npm install julia-system-one   # fetch during install
JULIA_MODEL_PATH=/opt/models/model.onnx              # or bring your own copy
JULIA_MODEL_CHUNKS_DIR=/opt/models/chunks            # or a directory of chunks
```

---

## 💻 Requirements

| | |
| :--- | :--- |
| **Node.js** | ≥ 18.17 |
| **Bun** | ≥ 1.0 |
| **Browsers** | The `wasm` backend |
| **OS** | Linux (glibc and musl/Alpine), macOS (arm64 and x64), Windows (x64 and arm64) |
| **Docker** | Debian, Ubuntu, Alpine |

No runtime dependencies. The right native binary for your machine is installed
automatically — nothing to compile, no system packages to add.

---

## 📥 What gets installed

The package itself is small; the heavy parts arrive as dependencies npm picks
for your platform, so you only download what you can run.

| | Size |
| :--- | ---: |
| `julia-system-one` (code, tokenizer, wasm engine) | ~8.5 MB |
| The one native binary for your platform | 8–26 MB |
| The 6 model chunks | ~148 MB total |
| The model on disk, after the first run | ~148 MB (INT8) |

The model is written next to the package when that directory is writable, and
to your user cache otherwise — so `npm i -g` and read-only containers work
without extra configuration. Every copy is checksum-verified, and a run that
is killed mid-download leaves nothing corrupt behind.

---

## 🛠️ Development

```bash
npm install
npm run check            # lint, unit, packaging, integration, e2e + install rehearsal
npm run check:quick      # the same minus the model-backed suites
npm run test:rehearsal   # install from a local registry and use it, Node + Bun
npm run bench            # measure on this machine
npm run lint             # syntax + packaging + docs consistency
```

CI builds every platform, proves each binary answers 10 questions, and uploads
the packages as artifacts. `verify-published.yml` installs a published version
from the real registry on every platform — Node and Bun, including Alpine for
musl — and runs a real inference.

---

## 🙏 Credits

**The model is [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) by
[Supersonic Labs](https://huggingface.co/SupersonicLabs).** The checkpoint, the
decision head, the training and the evaluation are theirs. This package only
makes it run where Python is not an option — Node.js, Bun and the browser.

| | |
| :--- | :--- |
| **Model** | [`SupersonicLabs/Julia-1`](https://huggingface.co/SupersonicLabs/Julia-1) — 144.3M parameters, a fine-tune of [`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small) |
| **ONNX build** | [`SupersonicLabs/Julia-1-ONNX`](https://huggingface.co/SupersonicLabs/Julia-1-ONNX) |
| **Encoder** | [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) (JHU CLSP) |
| **License** | Apache-2.0, for the model and for this package |

If you find this useful, the credit belongs upstream — star the model.

## 📄 License

[Apache-2.0](LICENSE) — the same license as the upstream project.

- **This package:** [Italo Almeida](https://github.com/italoalmeida0) —
  [julia-system-one](https://github.com/italoalmeida0/julia-system-one)
- **Model:** Supersonic Labs —
  [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)
