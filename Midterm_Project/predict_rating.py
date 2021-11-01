#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
import pickle
import numpy as np
from flask import Flask, request, jsonify


# In[ ]:


#creating a function for predicting
def predict_single(app, dv, model):
    X = dv.transform(app)
    y_pred = model.predict(X)
    rating = np.exp(y_pred)
    return rating


# In[ ]:


#loading model
with open('app_rating_model.pkl', 'rb') as f:
    dv, model = pickle.load(f)


# In[ ]:


app = Flask('rating')


# In[ ]:


#creating function for the web service
@app.route('/predict', methods=['POST'])
def predict():
    app = request.get_json()

    prediction = predict_single(app, dv, model)
    
    
    result = {
        'Rating': float(prediction),
    }

    return jsonify(result)


# In[ ]:


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

