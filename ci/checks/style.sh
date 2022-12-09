#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
##############################
# cuCollections Style Tester #
##############################

# Ignore errors and set path
set +e
PATH=/conda/bin:$PATH
# LC_ALL=C.UTF-8
# LANG=C.UTF-8

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
