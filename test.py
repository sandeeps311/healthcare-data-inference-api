import pickle

import joblib

filename = 'deployment/best_rf_model_pkl.sav'
loaded_model = pickle.load(open(filename, 'rb'))
# loaded_model = joblib.load(filename)
# result = loaded_model.predict(X_teststd)
