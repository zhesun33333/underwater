#!/usr/bin/env bash
set -euo pipefail

data_root="${VAST_DATA_ROOT:-/workspace}"
download_dir="${DATASET_DOWNLOAD_DIR:-$data_root/downloads/water}"
testset_dir="${TESTSET_DIR:-$data_root/testset_export}"
repo_id="bubbachuck3333/water"
revision="${DATASET_REVISION:-e398d8b6db7d2f9439daa2cc6319ed646ceb1bd3}"
filename="testset_export.tar.gz"
expected_size="2228680061"
expected_sha256="1824cb2b5c3c84a693d6ea1515bb2adf883ea29f55d0a43a6199eee6d894541a"
minimum_free_gib="${MINIMUM_FREE_GIB:-10}"

if [[ ! -d "$data_root" ]]; then
  echo "Vast data path does not exist: $data_root" >&2
  exit 2
fi
case "$download_dir" in
  "$data_root"/*) ;;
  *) echo "Refusing download directory outside $data_root: $download_dir" >&2; exit 2 ;;
esac
case "$testset_dir" in
  "$data_root"/*) ;;
  *) echo "Refusing test-set directory outside $data_root: $testset_dir" >&2; exit 2 ;;
esac

if [[ -f "$testset_dir/sft_test_highquality.jsonl" ]]; then
  echo "Existing extracted test set found; leaving it unchanged: $testset_dir"
  exit 0
fi
if [[ -e "$testset_dir" ]]; then
  echo "Refusing to overwrite incomplete existing path: $testset_dir" >&2
  exit 2
fi

available_kib="$(df -Pk "$data_root" | awk 'NR==2 {print $4}')"
required_kib="$((minimum_free_gib * 1024 * 1024))"
if (( available_kib < required_kib )); then
  echo "Less than ${minimum_free_gib} GiB is free on $data_root." >&2
  df -h "$data_root" >&2
  exit 2
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$data_root/.cache/huggingface}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY="${HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY:-1}"
export TMPDIR="${TMPDIR:-$data_root/.tmp}"
mkdir -p "$download_dir" "$HF_HOME" "$HF_XET_CACHE" "$TMPDIR"

echo "Dataset:    $repo_id"
echo "Revision:   $revision"
echo "Archive:    $download_dir/$filename"
echo "Extract to: $testset_dir"

# The dataset is currently public and ungated. If HF_TOKEN is present, the Hub
# client reads it from the environment; it is never passed as a CLI argument.
hf download "$repo_id" "$filename" \
  --repo-type dataset \
  --revision "$revision" \
  --local-dir "$download_dir" \
  --max-workers 1

archive="$download_dir/$filename"
test -f "$archive"
actual_size="$(wc -c < "$archive" | tr -d ' ')"
if [[ "$actual_size" != "$expected_size" ]]; then
  echo "Unexpected archive size: $actual_size bytes (expected $expected_size)." >&2
  exit 3
fi
actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "Unexpected archive SHA256: $actual_sha256" >&2
  echo "Expected: $expected_sha256" >&2
  exit 3
fi

if tar -tzf "$archive" | awk '
  BEGIN { unsafe = 0 }
  /(^\/|(^|\/)\.\.(\/|$))/ { unsafe = 1 }
  END { exit unsafe ? 0 : 1 }
'; then
  echo "Archive contains an unsafe path; refusing extraction." >&2
  exit 3
fi

staging="$(mktemp -d "$data_root/.water-extract.XXXXXX")"
cleanup() {
  case "$staging" in
    "$data_root"/.water-extract.*) rm -rf -- "$staging" ;;
    *) echo "Refusing unexpected cleanup path: $staging" >&2 ;;
  esac
}
trap cleanup EXIT

tar -xzf "$archive" -C "$staging"
candidate="$staging/testset_export"
mkdir -p "$(dirname "$testset_dir")"
mv "$candidate" "$testset_dir"

echo "Water test set ready: $testset_dir"
du -sh "$testset_dir" "$archive"
