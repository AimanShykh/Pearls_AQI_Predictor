name: Historical Backfill (manual)

on:
  workflow_dispatch:
    inputs:
      days:
        description: "Number of days of history to backfill"
        required: false
        default: "90"

jobs:
  backfill:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run backfill
        env:
          OWM_API_KEY: ${{ secrets.OWM_API_KEY }}
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
          HOPSWORKS_PROJECT: ${{ secrets.HOPSWORKS_PROJECT }}
        run: python src/backfill.py --days ${{ github.event.inputs.days }}

      - name: Upload parquet as artifact
        uses: actions/upload-artifact@v4
        with:
          name: backfill-data
          path: backfill_training_data.parquet
