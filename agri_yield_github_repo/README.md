# Agri Predictive Analytics: Early Crop-Yield Forecasting for Procurement

Forecast crop yield (quintal/ha) across six Vidarbha districts (Nagpur, Wardha, Amravati, Yavatmal, Chandrapur, Bhandara) early enough to plan procurement, and explain the drivers to business users.

> **All data in this repository is synthetic.** The project brief describes a fictional Nagpur agritech start-up and supplied no dataset, so `gen_data.py` simulates one (about 332,000 rows) with the problems the brief lists. The results show that the method works. They say nothing about real farms, and the model must be re-trained on real records before any operational use.

## Results at a glance

| Success criterion | Result | Verdict |
|---|---|---|
| Beat the baseline by at least 15% | 28.8% better on validation, 60.3% better on test | Met |
| Test performance within 5% of validation | +8.2% on the main metric (block-crop-year); +4.5% at plot level | **Not met** on the main metric |
| Top five drivers make business sense | Irrigation, cumulative rain, rain against crop water need, Jun-Sep rain, 3-year block-crop yield | Met |

The main metric is MAE in quintal/ha per block-crop-year, the unit a procurement plan uses. The gap criterion was missed and was not tuned away: validation is only two years, test is three, and the data drifts between periods (adversarial validation AUC 1.0). See [MODEL_CARD.md](MODEL_CARD.md) for the full discussion.

| Model | Validation MAE | Test MAE |
|---|---|---|
| Persistence (last year's yield, baseline) | 2.108 | 4.091 |
| 3-year mean | 2.311 | 2.996 |
| Ridge (per crop) | 1.994 | 2.405 |
| Random forest | 1.532 | 1.700 |
| LightGBM default | 1.538 | 1.713 |
| **LightGBM tuned (final)** | **1.500** | **1.623** |

Tuning improved cross-validated MAE only from 1.433 to 1.431, so the better test score of the tuned model over the default is probably noise.

![Evaluation](figures/evaluation.png)

## What the project covers

1. **Data cleaning.** Prices entered per quintal and per kg in one column, yields in wrong units (kg/ha, t/ha, quintal/acre), and 18.6% of monsoon rainfall readings missing. Cleaning was audited against a hidden ground-truth file that is never used as model input.
2. **Exploratory analysis and leakage audit.** Post-harvest columns (production, insurance claims, harvest price, late rain) cut error from 2.19 to 0.92 quintal/ha, so they were removed. A 300-row causality test confirms no later input changes a forecast.
3. **Time-aware split.** Train 2007-2019, validation 2020-2021, test 2022-2024. Cross-validation uses expanding windows inside the training years only.
4. **53 features,** 25 of them lag, rolling or ratio features, with past-years-only target encoding for villages and seed varieties.
5. **Models.** Baseline, ridge regression, random forest and LightGBM, compared with cross-validation. LightGBM was tuned with a 10-configuration random search.
6. **Business metrics.** MAE in quintal/ha, plus recall at fixed precision (at least 70%) for flagging a yield shortfall.
7. **Explainability.** SHAP values, partial dependence plots and one worked wrong prediction (Yavatmal wheat 2024).
8. **Fairness and stability.** Error and bias by district, crop, farm size and data completeness, per district-crop recalibration, and noise-robustness tests.
9. **Procurement plan for 2024.** 14 district-crop pairs to lock in early, 4 to hedge, 13 to buy opportunistically, 5 to monitor. In hindsight 32 of 36 calls were right.

![SHAP importance](figures/shap_bar.png)

## Repository layout

```
.
├── README.md
├── MODEL_CARD.md                  model card
├── agri_yield_model.ipynb         executed notebook with all outputs
├── gen_data.py                    synthetic data generator (fixed seed)
├── predict_yield.py               helper to score engineered features
├── requirements.txt
├── LICENSE
├── docs/
│   ├── project_report.docx        full written report
│   └── findings_deck.pptx         eight-slide findings deck
├── figures/                       all plots used in the notebook, report and deck
└── outputs/
    ├── yield_model.joblib         trained model bundle
    ├── metrics.json               every reported metric
    ├── procurement_priority_2024.csv
    ├── test_predictions.csv
    └── cv_results.csv, tuning_results.csv, shap_top_features.csv,
        fairness_before.csv, fairness_after.csv
```

The `data/` folder is not in the repository. It is about 39 MB and is recreated byte for byte from the fixed random seed.

## How to run

Tested on Python 3.12. Run every command from the repository root.

```bash
# 1. set up
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. generate the synthetic data (about 10 seconds, writes ./data)
python gen_data.py

# 3. open the notebook
jupyter notebook agri_yield_model.ipynb
```

The notebook generates the data itself if `data/` is missing. A full run takes about 20 minutes on a single CPU core, and the notebook rewrites the files in `figures/` and `outputs/`. Random seeds are fixed, but small numeric differences across library versions are possible. To check the pipeline quickly with smaller models and fewer tuning configurations, set `QUICK=1` before starting Jupyter (macOS and Linux: `QUICK=1 jupyter notebook`). Quick-mode numbers will differ from those above.

To execute without the interface:

```bash
jupyter nbconvert --to notebook --execute agri_yield_model.ipynb --output run.ipynb
```

## Using the trained model

```python
import predict_yield as p

bundle = p.load()                  # loads outputs/yield_model.joblib
print(bundle["features"])          # the 53 input columns
# yhat = p.predict(bundle, features_df)
```

`predict_yield.py` scores a table of already engineered features. Section 4 of the notebook shows how each feature is built from the raw tables, using only information available at the forecast date: 30 September for kharif crops, 31 January for rabi crops. The model predicts the log ratio of yield to the 3-year village-crop mean, then converts back to quintal/ha and applies the per district-crop recalibration factors.

## Limitations

- The data is simulated, so no claim about real yields, prices or farm behaviour follows from these results.
- The test-gap criterion was missed on the main metric, and the test period is only three years.
- Pest outbreaks and local shocks cannot be predicted from these features and cause the largest misses.
- Rainfall imputation error is large (about 127 mm), so forecasts are less reliable where gauges fail.
- Small district-crop cells, such as cotton in Bhandara (14-15 rows), are unreliable.

## License

MIT, see [LICENSE](LICENSE). Replace `YOUR NAME` in that file with your own name before publishing.
