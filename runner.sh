#!/bin/bash

# Function to read configuration from config.ini
read_config() {
  local config_file="config.ini"
  if [ -f "$config_file" ]; then
    echo "Loading configuration from $config_file..."
    force_precond=$(awk -F= '/^gen_precond_forced/ {print $2}' "$config_file" | tr -d ' ')
    debug_mode=$(awk -F= '/^debug/ {print $2}' "$config_file" | tr -d ' ')
    inject_error=$(awk -F= '/^inject_error/ {print $2}' "$config_file" | tr -d ' ')
    vocal_mode=$(awk -F= '/^vocal/ {print $2}' "$config_file" | tr -d ' ')
  else
    echo "ERROR: Config file $config_file not found!"
    exit
  fi

   # Check if the debug_mode variable is set and is equal to "true"
  if [[ $debug_mode == "true" ]]; then
    # Add "debug" to the MAKE_CONFIG variable
    MAKE_CONFIG="$MAKE_CONFIG -DENABLE_TESTS -g "
  fi

  # Check if the inject_error variable is set and is equal to "true"
  if [[ $inject_error == "true" ]]; then
    # Add "inject_error" to the MAKE_CONFIG variable
    MAKE_CONFIG="$MAKE_CONFIG -DINJECT_ERROR "
  fi
} 2> "./Build/build.log"

# Function to check if preconditioning is required
check_precond() {
  local mm_dir="test_subjects/mm"
  local precond_dir="test_subjects/precond"
  local precond_norm_dir="test_subjects/precond_norm"
  local precond_rcm_dir="test_subjects/precond_rcm"
  local force_precond="$1"

  # Check if the mm directory is empty
  if [[ -z $(find "$mm_dir" -mindepth 1 -type f -print) ]]; then
    echo "ERROR: No matrix files found in $mm_dir!"
    exit
  fi

  # Get the counts of matrices in different directories
  mm_count=$(find "$mm_dir" -type f | wc -l)
  precond_count=$(find "$precond_dir" -type f | wc -l)
  precond_norm_count=$(find test_subjects/rcm -type f | wc -l)
  precond_rcm_count=$(find "$precond_rcm_dir" -type f | wc -l)

  # Check if force_precond is set to true or if there are newer matrix files in mm
  if [[ "$force_precond" = true || $(find "$mm_dir" -type f -newer "$precond_dir" -print -quit ) || "$mm_count" -ne "$precond_count" || "$mm_count" -ne "$precond_norm_count" || "$mm_count" -ne "$precond_rcm_count" ]]; then
    echo "Precondition Generation is required or forced. (this will take a bit)"
    echo "Generating preconditioners..."
    (cd scripts; rm precond/* precond*/* norm/* rcm/* )
    if [[ $vocal_mode == "true" ]]; then
      (cd scripts; octave converter.m) 
    else
      (cd scripts; octave converter.m) > ./Build/build.log
    fi
  else
    echo "Precondition Generation is not required."
  fi
} 2>> "./Build/build.log"

# Function to build the project
build_project() {
  if [[ $vocal_mode == "true" ]]; then
    ( echo "Building the project..."; make CONFIG="$MAKE_CONFIG" ) | tee -a "./Build/build.log"
  else
    echo "Building the project..." >> "./Build/build.log"
    make CONFIG="$MAKE_CONFIG" >> "./Build/build.log" 2>&1
  fi
  MAKE_STATUS=$? 
}

# Function to run cgpc
run_cgpc() {
  if [[ $vocal_mode == "true" ]]; then
    echo "Running the project..."; 
    ( cd Build; ./cgpc ) #| tee "./Build/run.log"
  else
    echo "Running the project..."  | tee -a "./Build/run.log"
    ( cd Build; ./cgpc ) > "./Build/run.log" 2>&1
  fi
  RUN_STATUS=$? 
}

# Function to create the Build directory if it doesn't exist
create_build_dir() {
  local build_dir="Build"

  if [ ! -d "$build_dir" ]; then
    mkdir -p "$build_dir"
  fi
}

# Function to create the Data directory if it doesn't exist
create_data_dir() {
  local data_dir="Data"

  if [ ! -d "$data_dir" ]; then
    echo "Creating the Data directory..."
    mkdir -p "$data_dir"
  else
    echo "Data directory was found."
  fi
  echo "Build Complete! :)"
}  >> "./Build/build.log" 2>&1


# Function to perform additional actions
handle_data() {
  echo "Performing additional actions..."
  (cd Data; cat results_CCG_TEST.csv > combo.csv && sed '1d' results_CudaCG_TEST.csv >> combo.csv)  
  python3 scripts/mtx_table.py
  python3 scripts/gpu_percentages.py
  python3 scripts/iteration_graph.py
  python3 scripts/timings_graph.py
} > "./Build/run.log" 2>&1

echo "Starting the Build..."
# Main script
create_build_dir

echo "Reading config..."
# Read configuration from config.ini
read_config

echo "Making sure preconditioners are set..."
# Check if preconditioning is required
check_precond "$force_precond"

echo "Compiling the project..."
# Build the project
build_project
if [[ $MAKE_STATUS == 0 ]]; then
    echo "Build successful!"
else
    echo "ERROR : Make has failed! ($MAKE_STATS) Check the logs in Build!"
    exit
fi

# Create the Data directory if it doesn't exist
create_data_dir

# Run cgpc
run_cgpc
if [[ $RUN_STATUS == 0 ]]; then
    echo "Run successful!"
    echo "Preparing the data..."
    # Perform additional actions
    handle_data
    echo "Script completed successfully."
else
    echo "ERROR : Run has failed! ($RUN_STATUS) Check the logs in Build!"
    exit
fi
