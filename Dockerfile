# syntax=docker/dockerfile:1

FROM quay.io/jupyter/scipy-notebook:latest
RUN pip install --no-cache-dir matplotlib scikit-learn