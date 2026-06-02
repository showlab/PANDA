#!/usr/bin/env bash
# Install the optional vision inference stack into the selected Python environment.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash scripts/install_vision_tools.sh

Run after installing PANDA's core package. Activate a virtual environment first,
or select its interpreter with PANDA_PYTHON=/path/to/python (preferred) or PYTHON.
The default interpreter is python3. Standard pip environment variables apply.

Installs YOLOWorld, CLIP, and Real-ESRGAN inference dependencies. BasicSR 1.4.2
and its legacy companions are installed with --no-deps; stable dependencies are
installed explicitly instead of resolving their obsolete tb-nightly requirement.
This is an inference installation, not an upstream training environment. pip check
can still report missing tb-nightly metadata; stable tensorboard is not that package.
Model weights are not downloaded or bundled. Configure local model paths and
content hashes separately, as described in the README.
HELP
    exit 0
fi
if [[ $# -ne 0 ]]; then
    printf 'Unknown argument: %s (use --help)\n' "$1" >&2
    exit 2
fi

python_bin="${PANDA_PYTHON:-${PYTHON:-python3}}"
"$python_bin" -m pip --version
printf 'Installing vision inference dependencies with %s\n' "$python_bin"

# Install the stable runtime closure before bypassing legacy training metadata.
"$python_bin" -m pip install \
    'torch>=2.0' 'torchvision>=0.16' \
    'transformers>=4.45,<5' 'ultralytics>=8.3,<9' \
    'numpy>=1.26' 'opencv-python>=4.10' 'Pillow>=10.0' \
    'requests>=2.31' 'pyyaml>=6.0' scipy scikit-image \
    addict future lmdb tqdm yapf tensorboard filterpy numba \
    'clip @ https://github.com/openai/CLIP/archive/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.zip'
"$python_bin" -m pip install --no-deps \
    'basicsr==1.4.2' 'realesrgan==0.3.0' 'gfpgan==1.3.8' 'facexlib==0.3.0'

printf '%s\n' \
    'Vision inference packages installed; pretrained weights remain separate.' \
    'BasicSR compatibility is applied lazily by PANDA when super-resolution loads.' \
    'Upstream tb-nightly training metadata is intentionally unsatisfied; this is not a training install.'
