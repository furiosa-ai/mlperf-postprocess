SHELL := /bin/bash -o pipefail

.PHONY: install-deps
install-deps:
	@# a subset of dependencies from pyproject.toml, required for fresh venv
	@# (packaging is not a direct dependency, but is always installed through black)
	python -m pip install \
		"maturin[patchelf]~=1.1.0" \
		"ziglang==0.12.0.dev.168+67db26566" \
		isort \
		black \
		"packaging>=22.0"

.PYONY: lint
lint:
	cargo fmt --all --check \
	&& cargo -q clippy --release --all-targets -- -D rust_2018_idioms -D warnings

.PYONY: test
test:
	cargo test --release

.PHONY: build-wheels
build-wheels:
	@# workaround for https://github.com/rust-cross/cargo-zigbuild/pull/140
	BINDGEN_EXTRA_CLANG_ARGS= \
		maturin build --locked -r --zig --strip \
			-i python3.8 -i python3.9 -i python3.10 -i python3.11