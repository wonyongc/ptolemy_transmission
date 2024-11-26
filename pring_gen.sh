#!/bin/bash

# Usage: ./script.sh <number1> <number2> ... <numberN>
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <number1> <number2> ... <numberN>"
    exit 1
fi

MCS_FILE="LNGS_target.mcs"
MAKETEMPLATES_FILE="maketemplates.py"

for NUMBER in "$@"
do
    cp LNGS_target.mcs LNGS_target.mcs.orig
    cp maketemplates.py maketemplates.py.orig

    sed -i "s/pringphiplaceholder/$NUMBER/g" "$MCS_FILE"
    sed -i "s/pringphiplaceholder/$NUMBER/g" "$MAKETEMPLATES_FILE"

    python maketemplates.py della

    cp LNGS_target.mcs.orig LNGS_target.mcs
    cp maketemplates.py.orig maketemplates.py

done
