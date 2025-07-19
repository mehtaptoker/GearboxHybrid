#!/bin/bash

for i in {1..10}
do
  echo "Running test iteration $i"
  python -m unittest tests/test_environment.py
  if [ $? -ne 0 ]; then
    echo "Tests failed on iteration $i"
    exit 1
  fi
done

echo "All test iterations passed."
