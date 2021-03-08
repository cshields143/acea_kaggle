import pandas as pd
from .data import WaterbodyScaler

def calculate_errors(df):
    df['error'] = df['actual'] - df['predicted']
    df['error_abs'] = df['error'].abs()
    df['error_sq'] = df['error'] * df['error']
    return df

def train_test_and_score(mdl, X, y):

    # "train" like so:
    # - the first 2/3 of our data is always training data
    # - train the model on all available training data
    # - use this trained model to predict a single step into the future
    # - add that single step, with its actual value, into the training data
    # - retrain the model, predict the next step... etc

    # store all the results: actual/predicted, normalized/unnormalized
    true_scaled = []
    pred_scaled = []
    true_raw = []
    pred_raw = []

    # "sent" is our initial boundary between training/"testing" data
    # "_last_" is the final datapoint for which we can both predict an
    #     answer & grade that prediction
    sent = int(X.shape[0] * 0.67)
    _last_ = X.shape[0] - 1
    for i in range(sent, _last_):

        # how many iterations are left? for sanity purposes
        #print(f"!!!!! {_last_ - i + 1} !!!!!")

        # isolate training features & targets
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        Xy = pd.concat((X_train, y_train), axis=1)

        # normalize the data (we have to do this here, and not
        # somewhere else in the pipeline, because we don't
        # want our training data to be marred by knowledge of
        # all the future "testing" datapoints)
        ws = WaterbodyScaler(Xy)
        X_train_s, y_train_s = ws.scale(X_train), ws.scale(y_train)

        # get our single testing datapoint & normalize it
        # with the scaler that only saw training data
        y_true = y.iloc[i:i+1]
        y_true_s = ws.scale(y_true)

        # create a model, train it (ON NORMALIZED DATA), & get a prediction
        m = mdl()
        m.fit(X_train_s, y_train_s)
        y_pred_s = m.predict()

        # turn the prediction into a dataframe
        # (everything I'm doing here assumes there are column names...)
        y_pred_s = pd.DataFrame([y_pred_s], columns=y.columns)

        # UN-normalize our prediction
        y_pred = ws.unscale(y_pred_s)

        # record everything so we can analyze this data
        # (UN-dataframe everything... we don't need columns
        # anymore, just the straight values)
        true_scaled.append(y_true_s.values[0].tolist())
        pred_scaled.append(y_pred_s.values[0].tolist())
        true_raw.append(y_true.values[0].tolist())
        pred_raw.append(y_pred.values[0].tolist())

    # turn our recordings into dataframes, for convenience
    # (MAKE SURE TO RETAIN THE INDICES)
    idx = y.iloc[sent:_last_].index
    true_scaled = pd.DataFrame(true_scaled, index=idx, columns=y.columns)
    pred_scaled = pd.DataFrame(pred_scaled, index=idx, columns=y.columns)
    true_raw = pd.DataFrame(true_raw, index=idx, columns=y.columns)
    pred_raw = pd.DataFrame(pred_raw, index=idx, columns=y.columns)

    # NORMALIZED data needs to have error metrics calculated;
    # UNNORMALIZED data still needs to be processed a little
    # so that plotting is supes convenient
    metrics = {}
    history = {}
    for c in y.columns:

        # turn the truths & predictions for ONE COLUMN
        # into its own dataframe (both for scaled & unscaled)
        scaled = pd.concat({
            'actual': true_scaled[c],
            'predicted': pred_scaled[c]
        }, axis=1)
        raw = pd.concat({
            'actual': true_raw[c],
            'predicted': pred_raw[c]
        }, axis=1)

        # make sure we retain our indices
        scaled.index = true_scaled.index
        raw.index = true_raw.index

        # calculate absolute & squared errors for the column
        metrics[c] = calculate_errors(scaled)

        # save the raw values for plotting later
        history[c] = raw

    return metrics, history
