#!/bin/bash
# benchmark.sh — Run all cache experiments and save results
#
# Usage: ./src/benchmark.sh [output_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/../results}"

mkdir -p "$OUTPUT_DIR"

echo "=== Cache Experiments Benchmark Suite ==="
echo "Build dir: $BUILD_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check that binaries exist
for prog in cache_size cache_line false_sharing seq_vs_random matrix_traversal; do
    if [ ! -f "$BUILD_DIR/$prog" ]; then
        echo "Binary $BUILD_DIR/$prog not found. Run 'make' first."
        exit 1
    fi
done

run_and_save() {
    local name="$1"
    local binary="$BUILD_DIR/$2"
    local outfile="$OUTPUT_DIR/${name}.txt"

    echo "Running: $name"
    echo "--- $name ---" > "$outfile"
    echo "Date: $(date)" >> "$outfile"
    echo "" >> "$outfile"
    "$binary" 2>&1 | tee -a "$outfile"
    echo "" | tee -a "$outfile"
    echo "Saved to: $outfile"
    echo ""
}

run_and_save "01_cache_size"        "cache_size"
run_and_save "02_cache_line_size"   "cache_line"
run_and_save "03_false_sharing"     "false_sharing"
run_and_save "04_sequential_random" "seq_vs_random"
run_and_save "05_matrix_traversal" "matrix_traversal"

echo "=== All experiments complete ==="
echo "Results saved to: $OUTPUT_DIR"
