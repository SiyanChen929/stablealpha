#!/usr/bin/env bash
# One-command environment setup for StableAlpha (CSI500 factor mining).
# Usage: bash scripts/setup_env.sh [--with-qlib] [--skip-tests]
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_QLIB=0
RUN_TESTS=1
for arg in "$@"; do
  case "$arg" in
    --with-qlib) WITH_QLIB=1 ;;
    --skip-tests) RUN_TESTS=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), "Python >= 3.11 required"'

EXTRAS="dev,data-baostock,figures"
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv -p "$PY" .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -e ".[${EXTRAS}]" "lightgbm>=4.3"
  [ "$WITH_QLIB" = 1 ] && uv pip install "pyqlib>=0.9.6"
else
  [ -d .venv ] || "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -U pip
  python -m pip install -e ".[${EXTRAS}]" "lightgbm>=4.3"
  [ "$WITH_QLIB" = 1 ] && python -m pip install "pyqlib>=0.9.6"
fi

[ -f .env ] || cp .env.example .env
python -c "import cogalpha_plus, numpy, pandas, cvxpy, lightgbm; print('python', __import__('sys').version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| cvxpy', cvxpy.__version__, '| lightgbm', lightgbm.__version__)"
stablealpha check-config --project . >/dev/null && echo "check-config: ok"

if [ "$RUN_TESTS" = 1 ]; then
  pytest -o addopts="" -q -W ignore
fi
echo "Done. Activate with: source .venv/bin/activate"
