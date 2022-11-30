FROM quay.io/pypa/manylinux2014_x86_64

# Install procobuf compiler & python3-pip
RUN yum update -y \
    && yum install -y https://cbs.centos.org/kojifiles/packages/protobuf/3.6.1/4.el7/x86_64/protobuf-3.6.1-4.el7.x86_64.rpm \
    && yum install -y https://cbs.centos.org/kojifiles/packages/protobuf/3.6.1/4.el7/x86_64/protobuf-compiler-3.6.1-4.el7.x86_64.rpm \
    && yum install -y https://cbs.centos.org/kojifiles/packages/protobuf/3.6.1/4.el7/x86_64/protobuf-devel-3.6.1-4.el7.x86_64.rpm \
    && yum install -y python3-pip

# Install maturin
RUN pip3 install --upgrade pip && pip3 install --upgrade maturin

# Install rust
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH

# Set working directory, copy files
WORKDIR /app
COPY . .

# Add protos (make sure you have installed furiosa-libnux-extrinsic)
COPY /usr/share/furiosa /usr/share/furiosa
# COPY share/furiosa /usr/share/furiosa
