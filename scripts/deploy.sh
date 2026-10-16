#!/bin/bash
# Copies our code and job scripts to the cluster and records the git commit they come from
# (REVISION, printed at the start of every training job). Run from the repository root.
set -euo pipefail
rev=$(git rev-parse --short HEAD)
git diff --quiet HEAD -- src/camformer scripts || rev="$rev+uncommitted"
rsync -a --exclude=__pycache__ src/camformer/ bocconi-hpc:~/dl4cv/camformer/
rsync -a scripts/slurm/*.sh scripts/submit_chain.sh bocconi-hpc:~/dl4cv/
ssh bocconi-hpc "echo $rev > ~/dl4cv/camformer/REVISION"
echo "deployed $rev"
