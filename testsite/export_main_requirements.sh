#!/usr/bin/env bash
set -euo pipefail

# Always update requirements.txt next to this script, regardless of cwd.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_file="${script_dir}/requirements.txt"
temp_file="$(mktemp "${script_dir}/requirements.txt.XXXXXX")"
env_name="${1:-${CONDA_DEFAULT_ENV:-base}}"

cleanup() {
  rm -f -- "${temp_file}"
}
trap cleanup EXIT

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found in PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
  echo "Error: Conda environment '${env_name}' does not exist." >&2
  echo "Available environments:" >&2
  conda env list >&2
  exit 1
fi

conda run --no-capture-output -n "${env_name}" python -m pip freeze > "${temp_file}"
mv -- "${temp_file}" "${output_file}"
trap - EXIT

echo "Updated ${output_file} from Conda environment '${env_name}'."
