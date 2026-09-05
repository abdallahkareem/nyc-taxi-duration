```bash
#!/bin/bash

set -e

echo "=============================="
echo " NYC Taxi Duration ML Pipeline"
echo "=============================="

echo ""
echo "[1/3] Checking environment..."

if [ -z "$DATA_PATH" ]; then
    echo "ERROR: DATA_PATH is not set"
    exit 1
fi

echo "DATA_PATH=$DATA_PATH"

echo ""
echo "[2/3] Starting training..."

python src/main.py

echo ""
echo "[3/3] Pipeline completed successfully!"

echo ""
echo "Generated files:"
echo "----------------"
ls -lh models/
```
