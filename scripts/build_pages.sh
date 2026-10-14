#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
npm --prefix docs install
npm --prefix docs run docs:build

mkdir -p docs/.vitepress/dist
touch docs/.vitepress/dist/.nojekyll

if [ -f public/sonon.wasm ]; then
  cp -f public/sonon.wasm docs/.vitepress/dist/sonon.wasm
fi

if [ -f public/index.html ]; then
  cp -f public/index.html docs/.vitepress/dist/lab.html
fi

echo "Sonon documentation and web laboratory assembled in docs/.vitepress/dist"
