import pickle
from sklearn.metrics import mean_squared_error


def saveIfGood(model,X_test,y_test):
    filename = 'best_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred_new = loaded_model.predict(X_test)
    y_pred = model.predict(X_test)
    mse_new = mean_squared_error(y_test, y_pred_new)
    mse_best = mean_squared_error(y_test, y_pred)
    if mse_new < mse_best:
        filename = 'best_model.sav'
        pickle.dump(model, open(filename, 'wb'))

