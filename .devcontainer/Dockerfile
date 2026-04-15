FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ARG PANDOC_VERSION=3.8.3
ARG PANDOC_CROSSREF_VERSION=0.3.23

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/pandoc.deb \
        "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-1-amd64.deb" \
    && dpkg -i /tmp/pandoc.deb \
    && rm /tmp/pandoc.deb

RUN curl -fsSLo /tmp/pandoc-crossref.tar.xz \
        "https://github.com/lierdakil/pandoc-crossref/releases/download/v${PANDOC_CROSSREF_VERSION}/pandoc-crossref-Linux-X64.tar.xz" \
    && tar -xJf /tmp/pandoc-crossref.tar.xz -C /usr/local/bin pandoc-crossref \
    && chmod 0755 /usr/local/bin/pandoc-crossref \
    && rm /tmp/pandoc-crossref.tar.xz

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh
