#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "------------------------------------------------"
    echo "USAGE: extract_class_id.sh model_name_string"
    echo "Examples: "
    echo "extract_class_id.sh audi"
    echo "extract_class_id.sh SUV"
    echo "extract_class_id.sh Sedan"
    echo "------------------------------------------------"
    exit
fi
cat cars_class_id_class_name.txt | tail -196 | grep -i $1 | awk '{print $2}'
