import { Julia } from './agent.js';
import { env, isBrowser } from './env.js';

// Node builtins are imported lazily: this module is re-exported from the
// package entry point, so a static `import http from 'node:http'` makes the
// whole package unloadable in a browser - the import is resolved even when
// nothing calls serve().
let _node = null;
async function nodeBuiltins() {
  if (!_node) {
    const [http, fs, path, url] = await Promise.all([
      import('node:http'), import('node:fs'), import('node:path'), import('node:url')
    ]);
    _node = { http: http.default, fs: fs.default, path: path.default, url };
  }
  return _node;
}

/** The package's models/ directory: a path on Node, a URL in the browser. */
async function defaultModelDir() {
  if (isBrowser) return new URL('../models/', import.meta.url).href;
  const { path, url } = await nodeBuiltins();
  return path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), '..', 'models');
}

/**
 * What is actually loaded: the encoder named by rl_agent_config.json and the
 * sha256 of the model file. /health reports both, so a deployment can confirm
 * which checkpoint it serves instead of trusting a hardcoded name.
 *
 * Read lazily and cached: the sha256 streams the file once, not per request.
 */
let _identity = null;
async function modelIdentity(modelDir) {
  if (_identity) return _identity;
  const { fs, path, url } = await nodeBuiltins();
  const dir = modelDir || path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), '..', 'models');
  let encoder = 'unknown';
  let sha256 = 'unknown';
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(dir, 'rl_agent_config.json'), 'utf8'));
    if (cfg.encoder) encoder = cfg.encoder;
  } catch { /* keep unknown */ }
  try {
    const { createHash } = await import('node:crypto');
    const h = createHash('sha256');
    await new Promise((resolve, reject) => {
      const rs = fs.createReadStream(path.join(dir, 'model.onnx'));
      rs.on('data', (c) => h.update(c));
      rs.on('end', resolve);
      rs.on('error', reject);
    });
    sha256 = h.digest('hex');
  } catch { /* keep unknown */ }
  _identity = { encoder, sha256 };
  return _identity;
}

// Report the version we actually are, not a literal that goes stale: /health
// is what a deployment checks. Read lazily, so nothing touches the filesystem
// at module scope.
let _pkgVersion = null;
async function pkgVersion() {
  if (_pkgVersion) return _pkgVersion;
  try {
    const { fs, path, url } = await nodeBuiltins();
    const dir = path.dirname(url.fileURLToPath(import.meta.url));
    _pkgVersion = JSON.parse(fs.readFileSync(path.join(dir, '..', 'package.json'), 'utf8')).version;
  } catch {
    _pkgVersion = 'unknown';
  }
  return _pkgVersion;
}

/**
 * Start HTTP server exposing the TypeSafe Jev /v1/systemone wire protocol.
 * @param {Object} options - Server options:
 *   - host: string (default: '0.0.0.0' or process.env.HOST)
 *   - port: number (default: 8080 or process.env.PORT)
 *   - apiKey: string (optional, or process.env.JULIA_API_KEY / process.env.API_KEY)
 *   - julia: preloaded Julia instance (optional)
 *   - device: 'auto' | 'webgpu' | 'wasm' | 'cpu' (default: 'auto')
 * @returns {Promise<{ server: http.Server, url: string, julia: object, close: Function }>}
 */
export async function serve(options = {}) {
  if (isBrowser) {
    throw new Error('serve() needs an HTTP server, which a browser does not have. Use Julia.load() in the browser.');
  }
  const { http } = await nodeBuiltins();
  const host = options.host || env('HOST') || '0.0.0.0';
  // `port: 0` means "pick a free port" — it must not be treated as unset.
  const portRaw = options.port ?? env('PORT');
  const parsed = Number.parseInt(portRaw ?? '8080', 10);
  const port = Number.isFinite(parsed) ? parsed : 8080;
  const apiKey = options.apiKey || env('JULIA_API_KEY') || env('API_KEY') || null;
  // When we create the engine ourselves we also own its lifecycle: close()
  // must release it (the native backend spawns a julia-serve child process —
  // leaving it alive keeps the Node event loop busy and the process hangs
  // forever after the HTTP server is done).
  const ownsJulia = !options.julia;
  // The engine runs its own internal HTTP server (native backend). It must
  // never share the public port: on Windows two sockets CAN bind the same
  // address (SO_REUSEADDR), so requests would reach the wrong server and
  // shutdown would look broken. Keep it on a private loopback port.
  const julia = options.julia || (await Julia.load({ ...options, host: '127.0.0.1', port: 0 }));

  // End-to-end warmup: first real predict() pays tokenizer-cache fill +
  // any remaining lazy init. Do it once at startup (best-effort) so the
  // first Tetris piece doesn't eat the cold-start cost.
  if (!options.julia && options.warmup !== false) {
    try {
      await julia.predict('warmup', { w: { type: 'noul', instructions: 'warmup probe' } });
    } catch { /* best-effort */ }
  }

  const server = http.createServer(async (req, res) => {
    // Standard CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    // Handle preflight OPTIONS request
    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);

    // Healthcheck endpoint
    if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/health')) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      const identity = await modelIdentity(options.modelDir);
      res.end(JSON.stringify({
        status: 'ok',
        model: 'julia-1',
        encoder: identity.encoder,
        model_sha256: identity.sha256,
        version: await pkgVersion(),
        protocol: 'TypeSafe Jev /v1/systemone compatible'
      }));
      return;
    }

    // TypeSafe Jev evaluation endpoint
    if (req.method === 'POST' && url.pathname === '/v1/systemone') {
      // Optional bearer token authentication
      if (apiKey) {
        const authHeader = req.headers['authorization'] || '';
        if (authHeader !== `Bearer ${apiKey}`) {
          res.writeHead(401, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            error: 'Unauthorized: missing or invalid bearer token in Authorization header.'
          }));
          return;
        }
      }

      const chunks = [];
      let size = 0;
      req.on('data', chunk => {
        chunks.push(chunk);
        size += chunk.length;
        if (size > 4 * 1024 * 1024) { // 4MB guard
          res.writeHead(413, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Payload too large (max 4MB).' }));
          req.destroy();
        }
      });

      req.on('end', async () => {
        let payload;
        try {
          payload = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        } catch (e) {
          res.writeHead(422, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            error: 'Unprocessable Entity: invalid JSON payload.'
          }));
          return;
        }

        const state = payload.state;
        const questions = payload.questions;
        const requestedModel = payload.model || null;

        if (state === undefined || state === null) {
          res.writeHead(422, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            error: "Unprocessable Entity: missing required 'state' field."
          }));
          return;
        }

        if (!questions || typeof questions !== 'object' || Array.isArray(questions)) {
          res.writeHead(422, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            error: "Unprocessable Entity: 'questions' must be an object map of typed questions."
          }));
          return;
        }

        try {
          const result = await julia.predict(state, questions, requestedModel);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(result));
        } catch (err) {
          res.writeHead(422, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            error: `Evaluation failed: ${err.message || err}`
          }));
        }
      });
      return;
    }

    // 404 handler
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: `Not found: ${req.method} ${url.pathname}. Expected POST /v1/systemone`
    }));
  });

  // Keep-alive tuning: game clients (Tetris) fire one request per piece on
  // the same connection. Long keep-alive avoids TCP+handshake per move.
  server.keepAliveTimeout = 65000;
  server.headersTimeout = 66000;
  server.requestTimeout = 0;
  server.maxRequestsPerSocket = 0;

  return new Promise((resolve, reject) => {
    server.listen(port, host, () => {
      const displayHost = host === '0.0.0.0' ? 'localhost' : host;
      // use the port actually bound (0 means "any free port")
      const boundPort = server.address()?.port ?? port;
      const url = `http://${displayHost}:${boundPort}`;
      resolve({
        server,
        url,
        julia,
        close: async () => {
          await new Promise((resolve) => {
            server.close(() => resolve());
            // Node keeps idle keep-alive sockets open (undici pools them for
            // seconds), which would block server.close() — force them shut so
            // shutdown is immediate and deterministic.
            server.closeIdleConnections?.();
            server.closeAllConnections?.();
          });
          if (ownsJulia && typeof julia.close === 'function') await julia.close();
        }
      });
    });
    server.on('error', reject);
  });
}
