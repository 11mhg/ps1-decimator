#!/bin/sh
poetry run python -m pip download trimesh[easy] --dest .\wheels\ --only-binary=:all: --python-version=3.11 --platform=win_amd64