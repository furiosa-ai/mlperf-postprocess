FROM quay.io/pypa/manylinux2014_x86_64

# Install procobuf compiler & python3-pip
RUN yum update -y \
    && yum install -y python3-pip
#     && yum install -y https://cbs.centos.org/kojifiles/packages/protobuf/3.6.1/4.el7/x86_64/protobuf-3.6.1-4.el7.x86_64.rpm \
#     && yum install -y https://cbs.centos.org/kojifiles/packages/protobuf/3.6.1/4.el7/x86_64/protobuf-compiler-3.6.1-4.el7.x86_64.rpm \
#     && yum install -y https://cbs.centos.org/kojifiles/packages/protobuf/3.6.1/4.el7/x86_64/protobuf-devel-3.6.1-4.el7.x86_64.rpm \

# Install maturin
RUN pip3 install --upgrade pip setuptools wheel && pip3 install --upgrade maturin

# Install rust
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
RUN rustup toolchain install nightly-2023-02-01-x86_64-unknown-linux-gnu
RUN rustup component add rustfmt clippy --toolchain nightly-2023-02-01-x86_64-unknown-linux-gnu

# Set working directory
WORKDIR /app
