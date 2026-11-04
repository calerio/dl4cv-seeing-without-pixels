#!/bin/bash
# Copies our code and job scripts to the cluster and records the git commit they come from
# (REVISION, printed at the start of every training job). Run from the repository root.
set -euo pipefail
rev=$(git rev-parse --short HEAD)
# Uncommitted changes get a hash of the diff, so two different uncommitted deploys never share a revision.
git diff --quiet HEAD -- src/camformer scripts || rev="$rev+uncommitted-$(git diff HEAD -- src/camformer scripts | shasum | cut -c1-8)"
# -K keeps directory links on the cluster (data_files/ links to the upstream clone's CSVs; our own files sit
# next to those links) instead of replacing them with plain directories.
rsync -aK --exclude=__pycache__ src/camformer/ bocconi-hpc:~/dl4cv/camformer/
rsync -a scripts/slurm/*.sh scripts/submit_chain.sh bocconi-hpc:~/dl4cv/
ssh bocconi-hpc "echo $rev > ~/dl4cv/camformer/REVISION"
echo "deployed $rev"
