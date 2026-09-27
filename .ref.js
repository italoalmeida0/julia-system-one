
const TYPES = { choice: 0, score: 1, noul: 2 };

function encode(tokenizer, text) {
  return Array.from(tokenizer(text, { add_special_tokens: false }).input_ids.data, Number);
}

function pythonJSON(value) {
  if (Array.isArray(value)) return `[${value.map(pythonJSON).join(', ')}]`;
  if (value && value.constructor === Object)
    return `{${Object.entries(value).map(([key, item]) => `${JSON.stringify(key)}: ${pythonJSON(item)}`).join(', ')}}`;
  return JSON.stringify(value);
}

function displayProbabilities(values) {
  const winner = values.indexOf(Math.max(...values));
  if (values[winner] > 0.95 && values.every((value, i) => i === winner || value < 0.045))
    return values.map((_, i) => Number(i === winner));
  const visible = values.map(x => x >= 0.01 ? x : 0);
  const total = visible.reduce((a, b) => a + b, 0);
  return visible.map(x => x / total);
}

function serialize(tokenizer, row, maxLength, headLength, strict) {
  const type = row.type ?? 'choice';
  if (!(type in TYPES) || typeof row.question !== 'string' ||
      !(typeof row.state === 'string' || Array.isArray(row.state) || row.state?.constructor === Object) ||
      !Array.isArray(row.options) || row.options.length < 2 || row.options.length > 20 ||
      row.options.some(x => typeof x !== 'string' || !x) || (type === 'noul' && row.options.length !== 2)) {
    throw new TypeError('Invalid Julia decision request');
  }
  const state = typeof row.state === 'string' ? row.state : pythonJSON(row.state);
  const marker = tokenizer.mask_token;
  if (strict && [state, row.question, ...row.options].some(x => x.includes(marker)))
    throw new Error('Reserved model marker in request');
  const clean = x => x.replaceAll(marker, ' ');
  const head = encode(tokenizer, `${type} question: ${clean(row.question)}`);
  const optionIds = row.options.map(x => encode(tokenizer, ` ${clean(x)}`));
  if (strict && optionIds.some(x => x.length > 48)) throw new Error('Option exceeds 48-token model contract');
  let options = optionIds.map(x => [tokenizer.mask_token_id, ...x.slice(0, 48)]);
  let budget = headLength - options.reduce((sum, x) => sum + x.length, 0);
  if (budget < 16) {
    const perOption = Math.max(4, Math.floor((headLength - 16) / options.length));
    options = options.map(x => x.slice(0, perOption));
    budget = headLength - options.reduce((sum, x) => sum + x.length, 0);
  }
  if (strict && (head.length > budget || options.some((x, i) => x.length !== optionIds[i].length + 1)))
    throw new Error('Question/options exceed lossless head budget');
  const ids = [tokenizer.cls_token_id ?? tokenizer.bos_token_id, ...head.slice(0, Math.max(8, budget)), tokenizer.sep_token_id];
  const markers = [];
  for (const option of options) { markers.push(ids.length); ids.push(...option); }
  ids.push(tokenizer.sep_token_id);
  const stateIds = encode(tokenizer, clean(state));
  const room = maxLength - ids.length - 1;
  if (room < 1) throw new Error('Question/options exceed sequence budget');
  if (strict && stateIds.length > room) throw new Error('State exceeds lossless context budget');
  ids.push(...stateIds.slice(0, room), tokenizer.sep_token_id);
  return { ids, markers, qtype: TYPES[type] };
}

export class JuliaWebGPU {
  static async load(baseUrl = new URL('.', import.meta.url).href, options = {}) {
    const encoder = options.encoder ?? await this.loadWasmEncoder(baseUrl);
    const engine = await this.create({
      modelUrl: new URL('model.onnx', baseUrl).href,
      tokenizerUrl: baseUrl,
      encoder,
      ...options,
    });
    await engine.warm();
    return engine;
  }

  static async loadWasmEncoder(baseUrl) {
    const module = await import(new URL('wasm/julia_webgpu_encode.js', baseUrl).href);
    await module.default({ module_or_path: new URL('wasm/julia_webgpu_encode_bg.wasm', baseUrl) });
    const response = await fetch(new URL('tokenizer.json', baseUrl));
    if (!response.ok) throw new Error(`Tokenizer download failed: ${response.status}`);
    return new module.WasmEncoder(await response.text());
  }

  async warm() {
    await this.logits([{ state: '', question: 'Ready?', options: ['Yes', 'No'] }]);
    return this;
  }

  static async create({ modelUrl, tokenizerUrl, encoder = null, weightsUrl = `${modelUrl}.data`, maxLength = 1024, headLength = 256, strictEncoding = true }) {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable');
    if (headLength + 4 >= maxLength) throw new RangeError('headLength leaves no context');
    const ort = await import('onnxruntime-web/webgpu');
    const weightsPath = new URL(modelUrl, location.href).pathname.split('/').pop() + '.data';
    const tokenizerPromise = encoder ? Promise.resolve(null) : import('@huggingface/transformers')
      .then(({ AutoTokenizer }) => AutoTokenizer.from_pretrained(tokenizerUrl));
    const [session, tokenizer] = await Promise.all([
      ort.InferenceSession.create(modelUrl, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all', externalData: [{ path: weightsPath, data: weightsUrl }] }),
      tokenizerPromise,
    ]);
    return new JuliaWebGPU(ort, session, tokenizer, encoder, maxLength, headLength, strictEncoding);
  }

  constructor(ort, session, tokenizer, encoder, maxLength, headLength, strictEncoding) {
    Object.assign(this, { ort, session, tokenizer, encoder, maxLength, headLength, strictEncoding });
  }

  async logits(rows) {
    if (!rows.length) return [];
    const items = rows.map(row => this.encoder
      ? JSON.parse(this.encoder.encode(JSON.stringify(row), this.maxLength, this.headLength, this.strictEncoding))
      : serialize(this.tokenizer, row, this.maxLength, this.headLength, this.strictEncoding));
    const batch = items.length;
    const length = Math.ceil(Math.max(...items.map(x => x.ids.length)) / 8) * 8;
    const count = Math.max(...items.map(x => x.markers.length));
    const ids = new BigInt64Array(batch * length);
    const attention = new BigInt64Array(batch * length);
    const positions = new BigInt64Array(batch * count);
    const mask = new Uint8Array(batch * count);
    const qtype = new BigInt64Array(batch);
    items.forEach((item, i) => {
      item.ids.forEach((id, j) => { ids[i * length + j] = BigInt(id); attention[i * length + j] = 1n; });
      item.markers.forEach((pos, j) => { positions[i * count + j] = BigInt(pos); mask[i * count + j] = 1; });
      qtype[i] = BigInt(item.qtype);
    });
    const ort = this.ort;
    const output = await this.session.run({
      input_ids: new ort.Tensor('int64', ids, [batch, length]),
      attention_mask: new ort.Tensor('int64', attention, [batch, length]),
      marker_pos: new ort.Tensor('int64', positions, [batch, count]),
      marker_mask: new ort.Tensor('bool', mask, [batch, count]),
      qtype: new ort.Tensor('int64', qtype, [batch]),
    });
    const values = await output.logits.getData();
    return items.map((item, i) => Array.from(values.slice(i * count, i * count + item.markers.length)));
  }

  async predict(rows) {
    return (await this.logits(rows)).map(values => {
      const peak = Math.max(...values);
      const weights = values.map(x => Math.exp(x - peak));
      const total = weights.reduce((a, b) => a + b, 0);
      return { index: values.indexOf(peak), probabilities: displayProbabilities(weights.map(x => x / total)) };
    });
  }
}
