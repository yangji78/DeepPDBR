#!/bin/bash

mkdir -p /mnt/dataset/dockerfiles

echo "Extracting..."
tar -xJf /mnt/inputs/dataset.tar.xz -C /mnt/dataset/dockerfiles
echo "  + Done!"

find /mnt/dataset/dockerfiles -type f | sort \
  | python3 /app/app.py dataset
