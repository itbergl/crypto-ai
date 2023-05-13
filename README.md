# CITS4404 Project

## Setup

Install required packages.

```bash
pip install -r requirements.txt
```

Run `Optuna` study to optimise the hyperparameters.

```bash
python main.py Optuna
```

Run with the default hyperparameters which is the most optimal determined by the previous `Optuna` study conducted by us to find a candidate solution.

```bash
python main.py
```
