#!/bin/bash

ENTRY_POINT="."

echo "Enter the name of the final zip"
read name
zipify() {
  zip $1 -r "." -x '*.git*' '*__pycache__*'
}

zipify $name
