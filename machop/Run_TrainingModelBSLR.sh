#!/bin/bash

# Change the working directory to where ./ch is located
cd /home/laurie2905/mase/machop

# Define the absolute path to the output directory
OUTPUT_DIR="/home/laurie2905/mase/mase_output"

# Define your batch sizes and learning rates
BATCH_SIZES=(32 512)
LEARNING_RATES=(0.01 1e-6)

# Loop over each combination of batch size and learning rate
for BATCH_SIZE in "${BATCH_SIZES[@]}"
do
    for LR in "${LEARNING_RATES[@]}"
    do
        # Create a folder name based on the batch size and learning rate
        FOLDER_NAME="$OUTPUT_DIR/batch_LR"
        
        # Create the folder if it doesn't exist
        mkdir -p "$FOLDER_NAME"
        
        # Print the current configuration
        echo "Running model with Batch Size: $BATCH_SIZE, Learning Rate: $LR"
        
        # Run the training command with the current batch size and learning rate
        ./ch train jsc-tiny jsc --max-epochs 20 --batch-size $BATCH_SIZE --learning-rate $LR --project-dir "$FOLDER_NAME"
    done
done
