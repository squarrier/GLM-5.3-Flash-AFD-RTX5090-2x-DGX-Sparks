# The serving image: NVIDIA's PyTorch container plus the patched TensorFold tree's Python dependencies.
# The containers run the same tree from a read-only mount (PYTHONPATH=/tf/src), so a tree update needs no
# image rebuild. Build context: the patched tree (./build.sh images).
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
COPY . /opt/tensorfold-src
RUN python3 -m pip install --no-cache-dir /opt/tensorfold-src
