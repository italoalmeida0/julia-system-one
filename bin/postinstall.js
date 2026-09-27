#!/usr/bin/env node
/**
 * postinstall — cheap, silent-ish and never fatal.
 *
 * Since 1.1.0 the native binary and the model arrive as dependencies
 * (@julia-system-one/julia-serve-* and @julia-system-one/julia-model-chunk-*), so there is
 * nothing to download here. What still needs doing:
 *
 * 1. npm does not preserve the executable bit, so the binary inside the
 *    installed platform package would not run. Restore it.
 * 2. Optionally pre-assemble the model from the chunk packages when
 *    JULIA_PREFETCH_MODEL=1 (otherwise it happens lazily on first use, which
 *    keeps `npm install` fast).
 *
 * Everything is best-effort: a failure here must never break an install.
 * Silence the hint with JULIA_SKIP_MODEL_DOWNLOAD=1.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = path.join(__dirname, '..');
const MODELS_DIR = path.join(PKG_ROOT, 'models');

const QUIET = process.env.JULIA_SKIP_MODEL_DOWNLOAD === '1'
  || process.env.npm_config_ignore_scripts === 'true';

/** Every node_modules dir that could hold our @julia-system-one packages. */
function nodeModulesRoots() {
  const roots = [];
  let dir = __dirname;
  for (let i = 0; i < 8; i++) {
    roots.push(path.join(dir, 'node_modules'));
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return roots;
}

/** Restore +x on the binaries npm may have stripped. */
function restoreExecBits() {
  if (process.platform === 'win32') return 0;
  let fixed = 0;
  for (const root of nodeModulesRoots()) {
    const scope = path.join(root, '@julia-system-one');
    if (!fs.existsSync(scope)) continue;
    for (const name of fs.readdirSync(scope)) {
      if (!name.startsWith('julia-serve')) continue;
      const binDir = path.join(scope, name, 'bin');
      if (!fs.existsSync(binDir)) continue;
      for (const slot of fs.readdirSync(binDir)) {
        const slotDir = path.join(binDir, slot);
        if (!fs.statSync(slotDir).isDirectory()) continue;
        for (const file of fs.readdirSync(slotDir)) {
          if (file.endsWith('.exe')) continue;
          const p = path.join(slotDir, file);
          try {
            fs.chmodSync(p, 0o755);
            fixed++;
          } catch { /* best effort */ }
        }
      }
    }
  }
  return fixed;
}

async function prefetchModel() {
  const target = path.join(MODELS_DIR, 'model.onnx');
  if (fs.existsSync(target)) return 'already present';
  const { resolveModel } = await import(path.join(PKG_ROOT, 'src', 'model-resolver.js'));
  const { path: resolved, source } = await resolveModel({ modelDir: MODELS_DIR, quiet: true });
  return `${source} -> ${path.relative(PKG_ROOT, resolved)}`;
}

async function main() {
  const fixed = restoreExecBits();

  if (process.env.JULIA_PREFETCH_MODEL === '1') {
    try {
      const what = await prefetchModel();
      if (!QUIET) console.log(`[julia-system-one] model ready (${what})`);
    } catch (err) {
      if (!QUIET) {
        console.warn(`[julia-system-one] model prefetch failed (it will be fetched on first use): ${err.message}`);
      }
    }
  } else if (!QUIET && fixed === 0) {
    // most common case: nothing to say, but tell the user how it works once
    console.log('[julia-system-one] ready. The model is assembled from the @julia-system-one/julia-model-chunk-* packages on first use.');
  }
}

main().catch(() => { /* never fail an install */ });
