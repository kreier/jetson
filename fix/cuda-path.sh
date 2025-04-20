#!/bin/bash

set -eu

# Lines to check and add
LINE1='export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}'
LINE2='export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'

# File to modify
BASHRC="$HOME/.bashrc"

# Function to check and add a line
add_line_if_missing() {
    local line="$1"
    local file="$2"

    if ! grep -Fxq "$line" "$file"; then
        echo "$line" >> "$file"
        echo "Added: $line"
    else
        echo "Already exists: $line"
    fi
}

# Check both lines and add them if necessary
add_line_if_missing "$LINE1" "$BASHRC"
add_line_if_missing "$LINE2" "$BASHRC"

# Apply the changes to the current shell session
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Inform the user
echo "Changes applied! Both the ~/.bashrc file and the current shell session have been updated."
