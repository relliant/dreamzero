#!/bin/bash
# Generate DreamZero/GEAR metadata for every rx2_blackbox sortie sub-dataset.
#
# TWO MODES
#
# 1) In-place (default) -- writes new meta/*.json inside the source tree.
#      bash scripts/data/prepare_rx2_blackbox.sh --src <SRC_ROOT> [--force]
#
# 2) Mirror -- keeps SRC_ROOT read-only. Creates <OUT_ROOT>/<task>/<sub>/ with
#    symlinks for data/ and videos/, copies the original meta/ files, then
#    writes the DreamZero-specific meta files into the mirror only.
#      bash scripts/data/prepare_rx2_blackbox.sh \
#          --src /data/nas_ray/dataset/foundation_data/processed/lerobot/rx2_blackbox \
#          --out /data/nas_ray/home/siyu.luo/project/dreamzero_datasets/rx2_blackbox \
#          [--force]

set -euo pipefail

SRC_ROOT=""
OUT_ROOT=""
FORCE_FLAG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --src)   SRC_ROOT="$2"; shift 2 ;;
        --out)   OUT_ROOT="$2"; shift 2 ;;
        --force) FORCE_FLAG="--force"; shift ;;
        -h|--help)
            grep '^#' "$0" | head -20
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$SRC_ROOT" ]; then
    echo "ERROR: --src <RX2_BLACKBOX_ROOT> is required" >&2
    exit 2
fi
if [ ! -d "$SRC_ROOT" ]; then
    echo "ERROR: --src does not exist: $SRC_ROOT" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONVERTER="$REPO_ROOT/scripts/data/convert_rx2_blackbox_to_gear.py"
if [ ! -f "$CONVERTER" ]; then
    echo "ERROR: converter not found: $CONVERTER" >&2
    exit 1
fi

if [ -n "$OUT_ROOT" ]; then
    mkdir -p "$OUT_ROOT"
    echo "Mode: MIRROR"
    echo "  src: $SRC_ROOT (read-only)"
    echo "  out: $OUT_ROOT"
else
    echo "Mode: IN-PLACE (will write meta/*.json inside $SRC_ROOT)"
fi
echo

count=0
failed=0
for TASK_DIR in "$SRC_ROOT"/*/; do
    [ -d "$TASK_DIR" ] || continue
    task_name=$(basename "$TASK_DIR")
    for SUB in "$TASK_DIR"*/; do
        [ -d "$SUB/meta" ] || continue
        [ -d "$SUB/data" ] || continue
        [ -f "$SUB/meta/info.json" ] || continue
        sub_name=$(basename "$SUB")

        if [ -n "$OUT_ROOT" ]; then
            DEST="$OUT_ROOT/$task_name/$sub_name"
            mkdir -p "$DEST/meta"
            # symlink read-only sources
            ln -sfn "$SUB/data" "$DEST/data"
            ln -sfn "$SUB/videos" "$DEST/videos"
            # copy the original meta files (converter reads info.json,
            # reuses tasks.jsonl/episodes.jsonl; --force will overwrite
            # dreamzero-generated files it wrote before)
            for f in "$SUB/meta"/*; do
                [ -f "$f" ] || continue
                dest_f="$DEST/meta/$(basename "$f")"
                if [ ! -e "$dest_f" ] || [ -n "$FORCE_FLAG" ]; then
                    cp -f "$f" "$dest_f"
                fi
            done
            TARGET="$DEST"
        else
            TARGET="$SUB"
        fi

        echo "==> $TARGET"
        if python "$CONVERTER" --dataset-path "$TARGET" $FORCE_FLAG; then
            count=$((count + 1))
        else
            failed=$((failed + 1))
        fi
    done
done

echo
echo "Processed $count dataset(s). Failed: $failed."
[ "$failed" -eq 0 ]
