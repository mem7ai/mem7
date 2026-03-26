default:
  @just --list

build:
  maturin build --release

dev:
  maturin develop

clean:
  cargo clean
  rm -rf dist/ build/ *.egg-info/

openclaw-build:
  cd packages/openclaw-mem7 && npm run build

fmt:
  cargo fmt --all
  uv run --extra dev ruff format mem7 tests/python
  cd packages/openclaw-mem7 && npx prettier --write "src/**/*.ts"

fmt-check:
  cargo fmt --all --check
  uv run --extra dev ruff format --check mem7 tests/python
  cd packages/openclaw-mem7 && npm run fmt:check

clippy:
  cargo clippy --workspace --all-targets -- -D warnings

lint:
  uv run --extra dev ruff check mem7 tests/python
  cd packages/openclaw-mem7 && npm run lint

typecheck:
  cd packages/openclaw-mem7 && npx -p typescript tsc --noEmit

rust-test:
  cargo test --workspace

python-test:
  uv run --extra dev pytest tests/python -v

test: rust-test python-test

check: fmt-check clippy lint typecheck test
  @echo "All checks passed."
