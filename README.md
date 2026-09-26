# CardioLens

CardioLens is an educational Flask demo that uses a logistic regression model trained on the UCI Heart Disease dataset to return a heart disease screening flag from clinical measurements.

> **Research demo only:** This model has not been clinically validated. It is not a medical device, diagnosis, or treatment recommendation. Do not enter real patient information. Use synthetic or properly de-identified example data only.

## Run locally

```powershell
python -m pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:5000`.

## Deploy on Render Free

The repository includes `render.yaml`. In Render, choose **New → Blueprint**, connect the GitHub repository, and deploy the `cardiolens` web service. The Blueprint uses the included `.python-version`, installs `requirements.txt`, runs Gunicorn, and checks `/health`.

Render assigns a free `*.onrender.com` hostname and TLS certificate. A custom domain such as `cardiolens.in` requires registering and paying for that domain separately. Render Free services spin down after 15 minutes without inbound traffic; the first request after idle can take about a minute to wake the service. See [Render's free instance limits](https://render.com/docs/free).

## Model notes

- Target labels are binarized as `num == 0` (no disease) and `num > 0` (disease present in the source dataset).
- The demo uses a 30% model-score threshold to favor sensitivity on the fixed held-out split. On that split, sensitivity was 90.83% and specificity was 74.67%; these figures are not clinical validation and may not generalize.
- The displayed score is an uncalibrated model score. The screening flag must not be used to make clinical decisions.
- Retrain with `python model/train_model.py`. The script saves the model to the `model/` directory used by the app.

## Dataset attribution

The included `data/heart.csv` is based on the **Heart Disease** dataset from the UCI Machine Learning Repository (DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X)), created by Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano. The UCI dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). This project retains the required attribution; consult the UCI dataset page for the source and license terms.
