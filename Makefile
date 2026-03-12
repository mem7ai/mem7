.PHONY: build dev test test-rust test-python clean

build:
	maturin build --release

dev:
	maturin develop

test: test-rust test-python

test-rust:
	cargo test --workspace

test-python:
	python -m pytest tests/python -v

clean:
	cargo clean
	rm -rf dist/ build/ *.egg-info/

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

check:
	cargo check --workspace
