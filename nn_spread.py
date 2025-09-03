import os
import numpy as np
import pandas as pd

DATA_DIR = './'
P_SCORES = os.path.join(DATA_DIR, 'spreadspoke_scores.csv')
P_TEAMS = os.path.join(DATA_DIR, 'nfl_teams.csv')
P_WEEK1 = os.path.join(DATA_DIR, 'week_1_games.csv')
OUT_PRED = os.path.join(DATA_DIR, 'week1_nn_spread_predictions.csv')


def load_team_maps(p):
    t = pd.read_csv(p)
    name_col = 'team_name' if 'team_name' in t.columns else t.columns[0]
    abbr_col = 'team_id' if 'team_id' in t.columns else t.columns[1]
    full2abbr = {str(n).strip(): str(a).strip() for n, a in zip(t[name_col], t[abbr_col])}
    return full2abbr


def to_abbr(x, full2abbr):
    if pd.isna(x):
        return x
    s = str(x).strip()
    if s in full2abbr.values():
        return s
    return full2abbr.get(s, s)


def prepare_training(scores_csv, teams_csv, season_min=2000):
    full2abbr = load_team_maps(teams_csv)
    df = pd.read_csv(scores_csv)
    for c in ['score_home', 'score_away', 'schedule_season']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['team_home', 'team_away', 'score_home', 'score_away', 'schedule_season']).copy()
    df = df.loc[df['schedule_season'] >= season_min]
    df['home_abbr'] = df['team_home'].apply(lambda s: to_abbr(s, full2abbr))
    df['away_abbr'] = df['team_away'].apply(lambda s: to_abbr(s, full2abbr))
    df['margin'] = df['score_home'] - df['score_away']

    home_dummies = pd.get_dummies(df['home_abbr'], prefix='home')
    away_dummies = pd.get_dummies(df['away_abbr'], prefix='away')
    X_df = home_dummies.join(away_dummies)
    y = df['margin'].to_numpy()
    feature_cols = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=float)
    return X, y, feature_cols, full2abbr


def prepare_week1(week1_csv, feature_cols, full2abbr):
    wk = pd.read_csv(week1_csv)
    wk['home_abbr'] = wk['team_home'].apply(lambda s: to_abbr(s, full2abbr))
    wk['away_abbr'] = wk['team_away'].apply(lambda s: to_abbr(s, full2abbr))
    home_dummies = pd.get_dummies(wk['home_abbr'], prefix='home')
    away_dummies = pd.get_dummies(wk['away_abbr'], prefix='away')
    X_df = home_dummies.join(away_dummies)
    X_df = X_df.reindex(columns=feature_cols, fill_value=0)
    X = X_df.to_numpy(dtype=float)
    return wk, X


class SimpleNN:
    def __init__(self, input_dim, hidden_dim=32, lr=0.01, epochs=200, batch_size=64):
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(scale=0.01, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(scale=0.01, size=(hidden_dim, 1))
        self.b2 = np.zeros(1)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_grad(self, z):
        return (z > 0).astype(float)

    def fit(self, X, y):
        n = X.shape[0]
        y = y.reshape(-1, 1)
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                xb = X[batch]
                yb = y[batch]
                z1 = xb @ self.W1 + self.b1
                a1 = self._relu(z1)
                z2 = a1 @ self.W2 + self.b2
                err = z2 - yb
                dz2 = err / len(batch)
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * self._relu_grad(z1)
                dW1 = xb.T @ dz1
                db1 = dz1.sum(axis=0)
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return z2.ravel()


def main():
    X_train, y_train, feature_cols, full2abbr = prepare_training(P_SCORES, P_TEAMS)
    model = SimpleNN(X_train.shape[1], hidden_dim=64, lr=0.01, epochs=200, batch_size=128)
    model.fit(X_train, y_train)
    wk, X_week = prepare_week1(P_WEEK1, feature_cols, full2abbr)
    preds = model.predict(X_week)
    wk['predicted_margin_home'] = preds
    wk['predicted_favorite'] = np.where(preds >= 0, wk['home_abbr'], wk['away_abbr'])
    wk['predicted_spread'] = np.abs(preds)
    out_cols = ['schedule_date', 'schedule_season', 'schedule_week', 'team_home', 'team_away', 'predicted_favorite', 'predicted_spread', 'predicted_margin_home']
    wk[out_cols].to_csv(OUT_PRED, index=False)
    print('Wrote', OUT_PRED)


if __name__ == '__main__':
    main()
