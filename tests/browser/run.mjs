// A REAL browser test (headless Chromium), serving a real HTML page over HTTP.
//
// This exists because simulating a browser on Node does not work: it cannot
// catch module resolution, ESM over http, or the wasm compile limits. Running
// it for the first time found four bugs that the simulation had passed:
//
//   1. the entry point re-exports serve(), which imported node:http at module
//      scope - the browser could not load the package at all;
//   2. a relative modelDir was passed as a base to new URL(), which needs an
//      absolute one;
//   3. a relative wasmBase resolved against the module and produced
//      src/src/wasm-pkg/;
//   4. initSync compiles the 13 MB wasm on the calling thread, which Chrome
//      refuses above 8 MB - it needs the async init.
//
//   npm run test:browser
//
// Needs playwright and its chromium (npm i -D playwright && npx playwright
// install chromium). Serves the repo over HTTP and loads the real page.
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const MIME = { '.js':'text/javascript', '.mjs':'text/javascript', '.json':'application/json',
  '.wasm':'application/wasm', '.html':'text/html', '.onnx':'application/octet-stream' };

const srv = http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0].replace(/^\//, '')) || 'index.html';
  const p = path.join(ROOT, rel);
  if (!fs.existsSync(p) || !fs.statSync(p).isFile()) { res.writeHead(404); res.end('nope'); return; }
  res.writeHead(200, {
    'Content-Type': MIME[path.extname(p)] || 'application/octet-stream',
    'Access-Control-Allow-Origin': '*',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp'
  });
  fs.createReadStream(p).pipe(res);
});
await new Promise(r => srv.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${srv.address().port}/`;
console.log('servindo em', base);

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('console', m => { console.log('[browser:' + m.type() + ']', m.text().slice(0, 200)); if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', e => { console.log('[pageerror]', e.message.slice(0, 200)); errors.push(e.message); });
page.on('requestfailed', r => console.log('[req failed]', r.url().slice(0, 100), r.failure()?.errorText));

// a real page, so the origin is the server and imports are same-origin
await page.goto(base + 'tests/browser/page.html', { waitUntil: 'load' });

let result;
try {
  await page.waitForFunction(() => window.__result && (window.__result.ok || window.__result.error), null, { timeout: 240000 });
  result = await page.evaluate(() => window.__result);
} catch (e) {
  const partial = await page.evaluate(() => window.__result).catch(() => null);
  result = { ok: false, error: 'TIMEOUT', partial, consoleErrors: errors.slice(0, 5) };
}
console.log('RESULTADO:', JSON.stringify(result));
await browser.close();
srv.close();
process.exit(result.ok ? 0 : 1);
