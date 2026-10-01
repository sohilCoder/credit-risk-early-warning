#!/usr/bin/env bash
# End-to-end pipeline runner.
set -euo pipefail

echo "============================================"
echo " 1. Generating synthetic data"
echo "============================================"
python3 src/generate_synthetic_data.py --rows 500000

echo
echo "============================================"
echo " 2. Feature engineering (pandas mode)"
echo "     For distributed / production use:"
echo "     spark-submit src/feature_engineering_pyspark.py"
echo "============================================"
python3 src/feature_engineering_pandas.py

echo
echo "============================================"
echo " 3. Training models"
echo "============================================"
python3 src/train_model.py

echo
echo "============================================"
echo " 4. Drift monitoring"
echo "============================================"
python3 src/monitor_drift.py

echo
echo "============================================"
echo " 5. Stress scenarios"
echo "============================================"
python3 src/stress_scenarios.py

echo
echo "============================================"
echo " 6. Dashboard"
echo "============================================"
python3 src/build_dashboard.py

echo
echo "Done. Deliverables in outputs/"
