import pickle
from flask import Flask, jsonify, request
from predict_service import predict_lr, predict_svm, predict_dt, predict_knn

app = Flask('iris-predict')

types = {
    0: 'Iris Setosa',
    1: 'Iris Versicolour',
    2: 'Iris Virginica'
}

with open('models/logistic_regression.pck', 'rb') as f:
    sc, model = pickle.load(f)
with open('models/svm.pck', 'rb') as f:
    sc, model = pickle.load(f)
with open('models/decision_tree.pck', 'rb') as f:
    sc, model = pickle.load(f)
with open('models/knn.pck', 'rb') as f:
    sc, model = pickle.load(f)

@app.route('/lr_predict', methods=['POST'])
def lr_predict():
    data = request.get_json()
    flower_type, probability = predict_lr(data, sc, model)
    
    result = {
        'Flower': types[flower_type],
        'Probability' : round(probability*100, 2),
    }

    return jsonify(result)

@app.route('/svm_predict', methods=['POST'])
def svm_predict():
    data = request.get_json()

    flower_type, probability = predict_svm(data, sc, model)
    
    result = {
        'Flower': types[flower_type],
        'Probability' : round(probability*100, 2),
    }

    return jsonify(result)

@app.route('/dt_predict', methods=['POST'])
def dt_predict():
    data = request.get_json()

    flower_type, probability = predict_dt(data, sc, model)
    
    result = {
        'Flower': types[flower_type],
        'Probability' : round(probability*100, 2),
    }

    return jsonify(result)
@app.route('/knn_predict', methods=['POST'])
def knn_predict():
    data = request.get_json()

    flower_type, probability = predict_knn(data, sc, model)
    
    result = {
        'Flower': types[flower_type],
        'Probability' : round(probability*100, 2),
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)