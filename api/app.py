from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Carregando os modelos
model_rf_file = os.path.join(project_dir, 'models', 'random_forest_model.pkl')
model_lr_file = os.path.join(project_dir, 'models', 'logistic_regression_model.pkl')

model_rf = joblib.load(model_rf_file)
model_lr = joblib.load(model_lr_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    feature1 = data['feature1']
    feature2 = data['feature2']

    # Fazendo as previsões com os modelos
    prediction_rf = model_rf.predict([[feature1, feature2]])
    prediction_lr = model_lr.predict([[feature1, feature2]])

    response = {
        'model_rf_prediction': int(prediction_rf[0]),
        'model_lr_prediction': int(prediction_lr[0])
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()
