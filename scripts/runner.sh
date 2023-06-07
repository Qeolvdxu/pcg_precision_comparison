#!/bin/bash

all_empty=true

# Exclude the mm directory from the find command
if [[ -z $(find "../test_subjects" -mindepth 1 -type d ! -name "mm" -exec sh -c 'find "$0" -mindepth 1 -type f -o -type d | grep -q .' {} \; -print 2>/dev/null) ]]; then
    echo "All directories in test_subjects, except mm, are empty."
else
    all_empty=false
fi

if [ "$all_empty" = false ]; then
   echo "Do you want to generate new preconditioners?"
   read -rp "(this is only necessary if you changed matrices) [Y/n] " choice
   if [[ "$choice" =~ ^[Yy]$ ]]; then
      rm "../test_subjects/norm/*" 2>/dev/null
      rm "../test_subjects/rcm/*" 2>/dev/null
      rm "../test_subjects/precond*/*" 2>/dev/null
   fi
else
   echo "No CSR matrices found, generating files..."
fi
choice=${choice:-Y}

if [[ "$choice" =~ ^[Yy]$ ]]; then
    octave converter.m
else
    echo Using whatever is in the files.
fi

cd ../src
mkdir build
make

cd build
./cgpc N Y 1e-7 100000

cat results_CCG_TEST.csv > combo.csv
cat results_CudaCG_TEST.csv >> combo.csv
