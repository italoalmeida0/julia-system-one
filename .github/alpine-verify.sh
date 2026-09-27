#!/bin/sh
# alpine-verify.sh — install the published package on real Alpine and use it.
#
# Runs INSIDE the alpine container (see verify-published.yml). It lives in a
# file rather than inline in the workflow because the work needs three levels
# of quoting - YAML, the host shell, and the container shell - and nesting
# them produced unterminated strings twice.
#
# Env:
#   VERSION  the published version to install (e.g. 1.0.0)
#   ARCH     x64 or arm64
set -eu

echo "--- musl runtime ---"
ls /lib/ld-musl* || true

apk add --no-cache nodejs npm curl bash unzip

mkdir -p /verify && cd /verify
npm init -y >/dev/null
npm install "julia-system-one@$VERSION" --no-audit --no-fund

echo "--- installed @sys-one packages ---"
ls node_modules/@sys-one/ | sort

# Exactly the musl package for this arch, and nothing else specific: the whole
# point of the os/cpu split is that one machine gets one binary.
SPECIFIC=$(ls node_modules/@sys-one/ | grep '^julia-serve-' | grep -v universal || true)
echo "specific package: $SPECIFIC"
if [ "$SPECIFIC" != "julia-serve-linux-$ARCH" ]; then
  echo "expected julia-serve-linux-$ARCH, got $SPECIFIC"
  exit 1
fi

# The musl build ships as a self-extracting bundle (binary + libs in one file).
BUNDLE="node_modules/@sys-one/julia-serve-linux-$ARCH/bin/linux-$ARCH-musl/julia-serve.bundle"
if [ ! -f "$BUNDLE" ]; then
  echo "no musl bundle at $BUNDLE"
  find node_modules/@sys-one -type f | head -20
  exit 1
fi
chmod +x "$BUNDLE"
echo "--- bundle runs on bare Alpine ---"
"$BUNDLE" --help 2>&1 | head -3

# --- run it, with Node -----------------------------------------------------
cat > check.mjs <<'JS'
import { Julia } from 'julia-system-one';
const q = {
  department: {
    type: 'choice',
    instructions: 'Which department should handle this?',
    criteria: { billing: 'refunds and invoices', tech: 'bugs and crashes', sales: 'upgrades and contracts' }
  }
};
const cases = [
  'I was charged the wrong amount on my last invoice.',
  'We were billed twice on the March invoice and want a refund.',
  'The application crashes with a segfault when I open the settings page.',
  'The app freezes and throws an exception on startup.',
  'Your service has been down for six hours and nobody answers.'
];
const valid = new Set(Object.keys(q.department.criteria));
const t0 = Date.now();
const julia = await Julia.load({ backend: 'native' });
const loadMs = Date.now() - t0;
let pass = 0;
for (const prompt of cases) {
  const a = (await julia.predict(prompt, q)).answers.department;
  const sum = Object.values(a.probabilities || {}).reduce((x, y) => x + y, 0);
  if (valid.has(a.choice) && Math.abs(sum - 1) < 0.02) pass++;
  else console.log('  malformed', JSON.stringify({ prompt, got: a.choice, sum }));
}
await julia.close();
console.log(`well-formed: ${pass}/${cases.length} (load ${loadMs}ms)`);
if (pass < cases.length) process.exit(1);
JS

echo "--- Node on musl ---"
node check.mjs

# --- and with Bun ----------------------------------------------------------
echo "--- Bun on musl ---"
curl -fsSL https://bun.sh/install | bash
export BUN_INSTALL="${HOME}/.bun"
export PATH="${BUN_INSTALL}/bin:${PATH}"
bun --version
bun check.mjs

echo "MUSL VERIFIED ($ARCH)"
