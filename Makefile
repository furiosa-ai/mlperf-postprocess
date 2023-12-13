SHELL := /bin/bash -o pipefail
# as if maturin will do by default, can be overriden
WHEEL_DIR ?= ./wheels

.PHONY: install-deps
install-deps:
	python -m pip install \
		"maturin[patchelf]~=1.2.0" \
		"ziglang==0.12.0.dev.168+67db26566"

.PYONY: lint
lint:
	cargo fmt --all --check \
	&& cargo -q clippy --release --all-targets -- -D rust_2018_idioms -D warnings

.PYONY: test
test:
	cargo test --release

.PHONY: clean-wheels
clean-wheels:
	-rm -rf "${WHEEL_DIR}"/furiosa_native_postprocess*.whl

.PHONY: build-wheels
build-wheels: clean-wheels
	maturin build --locked -r --zig --strip -o "${WHEEL_DIR}" \
		-i python3.8 -i python3.9 -i python3.10 -i python3.11

.PHONY: check-dev-version
check-dev-version:
	@# Check that the version string contains `-dev` (i.e. is a dev version)
	cargo pkgid | grep -q '[-]dev'

.PHONY: publish-wheels-unchecked
publish-wheels-unchecked:
	maturin upload -r internal "${WHEEL_DIR}"/furiosa_native_postprocess*.whl

.PHONY: publish-wheels
publish-wheels: check-dev-version publish-wheels-unchecked
