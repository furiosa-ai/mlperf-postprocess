SHELL=/bin/bash -o pipefail

.PHONY: toolchain lint test

toolchain:

lint:
	cargo fmt --all --check && cargo -q clippy --all-targets -- -D rust_2018_idioms -D warnings

test:
	cargo test
