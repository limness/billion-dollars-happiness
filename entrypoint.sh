#!/bin/sh

set -eu

streamlit run client/main.py --server.port $PORT
