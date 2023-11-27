FROM quay.io/pypa/manylinux2014_x86_64

# Install procobuf compiler & python3-pip
RUN yum update -y \
    && yum install -y python3-pip

# Install maturin
RUN pip3 install --upgrade pip setuptools wheel && pip3 install --upgrade maturin

# Install rust
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
RUN rustup toolchain install nightly-2023-02-01-x86_64-unknown-linux-gnu
RUN rustup component add rustfmt clippy --toolchain nightly-2023-02-01-x86_64-unknown-linux-gnu

# Set working directory
WORKDIR /app
