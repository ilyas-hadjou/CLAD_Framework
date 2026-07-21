#!/usr/bin/env bash
# Download the full raw datasets used by CLAD from Loghub (Zenodo record 8196385).
# The Loghub-2k evaluation samples are already bundled under data/loghub-2k/.
#
# Full datasets are only needed for:
#   - the real-time streaming pipelines  (realtime/BGL, realtime/Thunderbird)
#   - the scalability experiment         (parser/eval/scalability.py)
#   - retraining the classifiers from scratch
#
# License: the Loghub datasets are freely available for research/academic use;
# cite the Loghub paper and refer to https://github.com/logpai/loghub
set -euo pipefail

ZENODO="https://zenodo.org/records/8196385/files"
DEST="$(dirname "$0")/full"
mkdir -p "$DEST"
cd "$DEST"

download_bgl() {
    echo ">>> BGL (57.5 MB zip, ~709 MB unpacked)"
    wget -c "$ZENODO/BGL.zip"
    unzip -o BGL.zip
}

download_thunderbird() {
    echo ">>> Thunderbird (2.0 GB tar.gz, ~30 GB unpacked)"
    wget -c "$ZENODO/Thunderbird.tar.gz"
    tar -xzf Thunderbird.tar.gz
}

download_hdfs() {
    echo ">>> HDFS_v1 (186.6 MB zip, includes anomaly labels)"
    wget -c "$ZENODO/HDFS_v1.zip"
    unzip -o HDFS_v1.zip
}

case "${1:-all}" in
    bgl)         download_bgl ;;
    thunderbird) download_thunderbird ;;
    hdfs)        download_hdfs ;;
    all)         download_bgl; download_thunderbird; download_hdfs ;;
    *) echo "usage: $0 [bgl|thunderbird|hdfs|all]"; exit 1 ;;
esac

echo "Done. Raw logs are under data/full/"
