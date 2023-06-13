#!/bin/bash

set -e

all_empty=true

# Exclude the mm directory from the find command
if [[ -z $(find "test_subjects" -mindepth 1 -type d ! -name "mm" -exec sh -c 'find "$0" -mindepth 1 -type f -o -type d | grep -q .' {} \; -print 2>/dev/null) ]]; then
    echo "All directories in test_subjects, except mm, are empty."
else
    all_empty=false
fi

# if there are already matrccies douns
if [ "$all_empty" = false ]; then
   echo "Do you want to generate new CSR and Preconditioners?"
   read -rp "(this is only necessary if you changed matrices) [Y/n] " choice
   if [[ "$choice" =~ ^[Yy]$ ]]; then
      rm ../test_subjects/norm/* 2>/dev/null
      rm ../test_subjects/rcm/* 2>/dev/null
      rm ../test_subjects/precond*/* 2>/dev/null
   fi
else
   echo "No CSR/Preconditioner matrix files found! Will generate new ones for you!"
fi
choice=${choice:-Y}

# Preconditioner matrices option
read -rp "Use preconditioner matrices? [Y/N] (default: Y): " precond_choice
precond_choice=${precond_choice:-Y}

# CPU and GPU concurrent execution option
read -rp "Run CPU and GPU concurrently? [Y/N] (default: Y): " concurrent_choice
concurrent_choice=${concurrent_choice:-Y}

# Convergence tolerance
read -rp "Convergence tolerance (default: 1e-7): " tolerance
tolerance=${tolerance:-1e-7}

# Iteration cap
read -rp "Iteration cap (default: 1000): " iteration_cap
iteration_cap=${iteration_cap:-1000}

echo "Select the flags for the make command:"

# GPU Mode
read -rp "GPU Mode (debug/release): " gpu_mode
gpu_mode=${gpu_mode:-release}

# CPU Mode
read -rp "CPU Mode (debug/release): " cpu_mode
cpu_mode=${cpu_mode:-release}

# GPU Precision
read -rp "GPU Precision (single/double): " gpu_preci
gpu_preci=${gpu_preci:-single}

# CPU Precision
read -rp "CPU Precision (single/double): " cpu_preci
cpu_preci=${cpu_preci:-double}

if [[ "$choice" =~ ^[Yy]$ ]]; then
    (cd scripts; octave converter.m)
else
    echo Using whatever is in the files.
fi

make cpu_mode="$cpu_mode" gpu_mode="$gpu_mode" gpu_preci="$gpu_preci" cpu_preci="$cpu_preci"

echo "running the CG"
(cd Build; ./cgpc "$precond_choice" "$concurrent_choice" "$tolerance" "$iteration_cap")

echo "Creating the Data"
(cd Build; cat results_CCG_TEST.csv > combo.csv && cat results_CudaCG_TEST.csv >> combo.csv)
mkdir -p Data 2>/dev/null
mv Build/*.csv Data/.

python3 scripts/gpu_percentages.py
python3 scripts/iteration_graph.py
python3 scripts/timings_graph.py
