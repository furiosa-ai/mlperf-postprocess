SHELL=/bin/bash -o pipefail

.PHONY: toolchain lint test

check-docker-tag:
ifndef DOCKER_TAG
	$(error "DOCKER_TAG is not set")
endif

toolchain:

lint:
	cargo fmt --all --check && cargo -q clippy --release --features cpp_impl --all-targets -- -D rust_2018_idioms -D warnings

test:
	cargo test --release --features cpp_impl

docker-build: check-docker-tag
	DOCKER_BUILDKIT=1 docker build -t asia-northeast3-docker.pkg.dev/next-gen-infra/furiosa-ai/mlperf-postprocess:${DOCKER_TAG} --secret id=furiosa.conf,src=/etc/apt/auth.conf.d/furiosa.conf -f docker/Dockerfile ./docker/

docker-push: check-docker-tag
	docker push asia-northeast3-docker.pkg.dev/next-gen-infra/furiosa-ai/mlperf-postprocess:${DOCKER_TAG}
