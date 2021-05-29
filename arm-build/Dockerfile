FROM debian:10.9

# common packages
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    ca-certificates curl file \
    build-essential pkg-config \
    autoconf automake autotools-dev libtool xutils-dev \
    gcc-arm-linux-gnueabihf libc6-dev-armhf-cross g++-arm-linux-gnueabihf


RUN dpkg --add-architecture armhf
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    libx11-dev:armhf libasound2-dev:armhf
    
# RUN rm -rf /var/lib/apt/lists/*

ENV SSL_VERSION=1.0.2u

RUN curl https://www.openssl.org/source/openssl-$SSL_VERSION.tar.gz -O && \
    tar -xzf openssl-$SSL_VERSION.tar.gz && \
    cd openssl-$SSL_VERSION && ./config && make depend && make install && \
    cd .. && rm -rf openssl-$SSL_VERSION*

ENV OPENSSL_LIB_DIR=/usr/local/ssl/lib \
    OPENSSL_INCLUDE_DIR=/usr/local/ssl/include \
    OPENSSL_STATIC=1

# install all 3 toolchains
RUN curl https://sh.rustup.rs -sSf | \
    sh -s -- --default-toolchain stable -y && \
    /root/.cargo/bin/rustup update beta && \
    /root/.cargo/bin/rustup update nightly

ENV PATH=/root/.cargo/bin:$PATH

RUN rustup target add armv7-unknown-linux-gnueabihf

WORKDIR /app

ADD Cargo.toml ./
ADD Cargo.lock ./
ADD arm-build/.cargo ./.cargo

ADD src ./src

ENV PKG_CONFIG_ALLOW_CROSS=1
ENV PKG_CONFIG_PATH=/usr/share/pkgconfig/
ENV PKG_CONFIG_PATH_armv7_unknown_linux_gnueabihf=/usr/lib/arm-linux-gnueabihf/pkgconfig

# RUN cargo build --target=armv7-unknown-linux-gnueabihf --release
