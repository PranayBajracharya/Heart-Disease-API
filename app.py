from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__)

CORS(app)

svm_modal = pickle.load(open('SVM.pkl', 'rb'))
decision_tree_modal = pickle.load(open('DecisionTree.pkl', 'rb'))

def sigmoid(x):
  return 1/(1 + np.exp(-x))


def SVM(input):
  preprocessed_input = np.array([input])
  predicted_class = svm_modal.predict(preprocessed_input)

  print(predicted_class[0])
  return predicted_class[0]


def decisionTree(input):
  preprocessed_input = np.array([input])
  predicted_class = decision_tree_modal.predict(preprocessed_input)

  print(predicted_class[0])
  return predicted_class[0]


def logisticRegression(input):
  B = -0.3089914854382788
  W = np.array([[ 0.09129111],[-6.33505535],[ 8.71898628],[-0.12598532],[-0.01500191],[ 0.24280076],[ 1.81233911],[ 0.17489223],[-5.41853826],[-6.05298511],[ 1.46312178],[-9.01810412],[-7.43117029]])

  Z = np.dot(W.T, input) + B

  prob = sigmoid(Z)
  print("prob", prob[0])
  if prob[0] > 0.5:
    prediction = 1
  else:
    prediction = 0
  return prediction  


@app.route('/', methods=['POST'])
def result():
  data = request.get_json()

  age = data['age']
  sex = data['sex']
  ca = data['ca']
  chol = data['chol']
  cp = data['cp']
  exang = data['exang']
  fbs = data['fbs']
  oldpeak = data['oldpeak']
  restecg = data['restecg']
  slope = data['slope']
  thalach = data['thalach']
  thal = data['thal']
  trestbps = data['trestbps']

  # input_data = (70,1,0,145,174,0,1,125,1,2.6,0,0,3)
  input_data = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)


  logistic_regression = logisticRegression(input_data)
  svm = SVM(input_data)
  dt = decisionTree(input_data)

  return jsonify({'logistic_regression': str(logistic_regression), 'svm': str(svm), 'dt': str(dt)})

if __name__ == '__main__':
  app.run(port=8000, debug=True)
