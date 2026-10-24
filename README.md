# Home Bill Tracker

Track your household bills and predict next month's amounts with machine learning. The live page runs the models in your browser, and the Python version does the same from the command line.

## Live Demo

Visit the tracker: [View Live Bill Tracker](https://kowshikreddy099.github.io/home-bill-tracker/)

## Features

- This month's total next to the ML estimate for next month
- A chart of the last 24 months for every bill, with the estimate on the end
- Four models compared on every bill: Repeat last month, 3-month average, Ridge regression and Boosted trees
- Walk-forward testing: each model is trained on earlier months and tested on the next one, and the one with the smallest error is used
- "Compare models" list on every bill card showing each model's average error
- Add the real bills each month and the tool predicts the month after
- Paste your own bill history as a table
- Light and dark mode, works on phones
- Data is saved in your own browser

## How the ML works

For every month, features are built from older months only:

| Feature | Meaning |
| --- | --- |
| lag 1, 2, 3 | bill amount 1, 2 and 3 months ago |
| 3-month average | average of those three |
| trend | month counter, to catch steady rises |
| month of year | sine and cosine, so December sits next to January and seasons are learned |

The models are checked with walk-forward validation. Starting from the 16th month, each model trains on all earlier months and predicts the next one. The average absolute error decides the winner for each bill. Fixed bills such as rent usually pick Repeat last month. Seasonal bills such as electricity usually pick Ridge regression.

## Project structure

```
.
├── index.html          # the live page, ML runs in the browser
├── ml/
│   ├── predict_bills.py          # Python version, uses scikit-learn
│   ├── generate_sample_data.py   # makes example data in bills.csv
│   ├── bills.csv                 # example data, replace with your own
│   └── requirements.txt
├── LICENSE
└── README.md
```

## Run Locally

### The web page

1. Clone this repository:
   ```bash
   git clone https://github.com/kowshikreddy099/home-bill-tracker.git
   ```
2. Open the project folder:
   ```bash
   cd home-bill-tracker
   ```
3. Open `index.html` in your browser, or use the Live Server extension in VS Code.

### The Python version

Requires Python 3.9 or newer.

```bash
cd ml
pip install -r requirements.txt
python predict_bills.py
```

On Mac or Linux use `pip3` and `python3`. The results are printed and saved to `next_month_prediction.csv`.

## Using your own bills

Use a table like this, one row per month, oldest first, no gaps:

```csv
month,rent,electricity,water,internet
2025-01,19000,742.50,225.00,899
2025-02,19000,690.00,218.40,899
```

- On the web page, paste it into **Your bill history** and press **Apply and re-predict**.
- For the Python version, save it as `ml/bills.csv`.
- You need 16 or more months for the ML models. With less history the tool uses the simple methods.

The data that ships with the project is generated, so its numbers show how the tool works and are not a real forecast.

## Tech stack

- Web page: HTML, CSS and JavaScript with no libraries. Ridge regression and gradient boosting are written from scratch
- Python version: Python, pandas, NumPy, scikit-learn

## Connect With Me

- GitHub: [kowshikreddy099](https://github.com/kowshikreddy099)

Built by Venkata Kowshik Reddy Dadani, Hyderabad, India.
