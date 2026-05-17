# Circuit Notebooks

The notebooks are the full narrative analyses for each circuit. They are written in English and preserve the circuit-specific interpretation used in the article.

| Notebook | Grand Prix | Circuit Focus |
|---|---|---|
| `Notebook_Bahrain.ipynb` | Bahrain Grand Prix | Bahrain International Circuit race pace in desert-night conditions |
| `Notebook_Saudi.ipynb` | Saudi Arabian Grand Prix | Jeddah Corniche Circuit race pace on a fast street circuit |
| `Notebook_USA.ipynb` | United States Grand Prix | Circuit of the Americas race pace under warmer afternoon conditions |
| `Notebook_Italia.ipynb` | Italian Grand Prix | Monza race pace on a low-drag, high-speed circuit |
| `Notebook_Hungary.ipynb` | Hungarian Grand Prix | Hungaroring race pace on a lower-speed, degradation-sensitive circuit |

Each notebook follows the same analysis flow: data preparation, exploratory diagnostics, RBF weather transformations, correlation/PCA checks, Linear Regression, XGBoost, sliding-window validation, sequential holdout, and COS metrics.

The notebooks mirror the current script methodology: `Year` is treated as a
numerical predictor, `Driver`, `Team`, and `pirelliCompound` are categorical
predictors, and PCA diagnostics exclude the supervised target
`LapTime_seconds`. Circuit-specific removals are also reflected in the notebook
feature lists: United States removes `TempDelta_RBF_Median` and `Year`, Saudi
Arabia removes `TrackTemp_RBF_Median`, and Hungary removes
`Humidity_RBF_Median`.

Generated notebook artifacts such as temporary plots and local parameter JSON files are ignored by Git.
