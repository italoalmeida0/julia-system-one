// The esm.sh path, end to end, in a real browser.
//
// The browser claim was only ever tested by serving files from disk, which
// never goes through a CDN. This runs the real thing in headless Chromium:
// import from esm.sh, load the wasm engine, fetch the model over HTTP, answer
// a question.
//
// It found two bugs that disk-serving could not: esm.sh rewrites every `node:`
// import into a browser shim and bundles server.js / julia-native.js /
// model-resolver.js into the entry (index.js re-exports serve), and those
// called fileURLToPath(import.meta.url) at module scope - which throws there
// before any code runs. And initSync compiles the ~13 MB wasm on the calling
// thread, which Chrome refuses above 8 MB.
//
//   npm run test:esmsh
//
// Needs playwright and its chromium:
//   npm i -D playwright && npx playwright install chromium
//
// Set LAYA_TARGET to test a published version instead of a commit:
//   LAYA_TARGET=julia-system-one@1.0.1 npm run test:esmsh
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const TARGET = process.env.LAYA_TARGET || 'gh/italoalmeida0/julia-system-one@main/src/index.js';
const MIME = { '.js':'text/javascript', '.mjs':'text/javascript', '.json':'application/json',
  '.wasm':'application/wasm', '.onnx':'application/octet-stream' };

// serve the model and the wasm engine; the code comes from esm.sh
const srv = http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0].replace(/^\//, ''));
  const p = path.join(ROOT, rel);
  if (!fs.existsSync(p) || !fs.statSync(p).isFile()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, {
    'Content-Type': MIME[path.extname(p)] || 'application/octet-stream',
    'Access-Control-Allow-Origin': '*'
  });
  fs.createReadStream(p).pipe(res);
});
await new Promise(r => srv.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${srv.address().port}/`;

const browser = await chromium.launch();
const page = await browser.newPage();
const logs = [];
page.on('console', m => logs.push(`[${m.type()}] ${m.text().slice(0, 200)}`));
page.on('pageerror', e => logs.push(`[pageerror] ${e.message.slice(0, 200)}`));

await page.route(base + 'page.html', (r) =>
  r.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><body></body>' }));
await page.goto(base + 'page.html');

const result = await page.evaluate(async ({ base, target }) => {
  const out = { steps: [] };
  try {
    const { Julia } = await import(`https://esm.sh/${target}`);
    out.steps.push('imported');
    const julia = await Julia.load({
      backend: 'wasm',
      modelDir: base + 'models/',
      wasmBase: base + 'src/wasm-pkg/'
    });
    out.steps.push('loaded');
    const r = await julia.predict('We were billed twice and want a refund.', {
      department: { type: 'choice', instructions: 'Which department?',
        criteria: { billing: 'refunds', tech: 'bugs', sales: 'upgrades' } }
    });
    out.answer = r.answers.department.choice;
    out.conf = r.answers.department.confidence;
    await julia.close();
  } catch (e) { out.error = String((e && e.message) || e); }
  return out;
}, { base, target: TARGET });

console.log('RESULTADO:', JSON.stringify(result, null, 1));
if (logs.length) { console.log('LOGS:'); for (const l of logs.slice(0, 6)) console.log(' ', l); }
await browser.close();
srv.close();
process.exit(result.answer ? 0 : 1);
