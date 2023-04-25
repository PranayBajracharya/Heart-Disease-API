from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pickle
import csv

app = Flask(__name__)

CORS(app)

MAX_TRESTBPS = 200
MIN_TRESTBPS = 94

MAX_CHOL = 564
MIN_CHOL = 126

MAX_THALACH = 202
MIN_THALACH = 71

svm_modal = pickle.load(open('SVM.pkl', 'rb'))
decision_tree_modal = pickle.load(open('DecisionTree.pkl', 'rb'))

def normalize(value, max, min):
  normalized_value = (value - min) / (max - min)
  if normalized_value > 1:
    normalized_value = 1
  if normalized_value < 0:
    normalized_value = 0
  return normalized_value

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
  B = 1.4529196202976526
  # W = np.array([[-0.01184615],[-1.63559982],[ 1.00394245],[-0.76612724],[-0.32547678],[-0.32404015],[ 0.63957675],[ 2.16091011],[-1.06465917],[-1.76134254],[ 0.66852324],[-0.86071214]])
  W = np.array([[-0.03054175],[-1.75988952],[ 0.87459045],[-0.88589462],[-0.65953262],[-0.19379372],[ 2.7781281 ],[-1.18520633]])

  Z = np.dot(W.T, input) + B

  prob = sigmoid(Z)
  print("prob", prob[0])
  if prob[0] > 0.5:
    prediction = 1
  else:
    prediction = 0
  return prediction  

# def accuracy(age, sex, cp, trestbps, chol, fbs, thalach, exang):
#   age =int(age)
#   sex =int(sex)
#   cp =int(cp)
#   trestbps =float(trestbps)
#   chol =float(chol)
#   fbs =int(fbs)
#   thalach =float(thalach)
#   exang = int(exang)
#   norm_bps = normalize(trestbps, MAX_TRESTBPS, MIN_TRESTBPS)
#   norm_chol = normalize(chol, MAX_CHOL, MIN_CHOL)
#   norm_thalach = normalize(thalach, MAX_THALACH, MIN_THALACH)

#   # input_data = (60,1,0,145,174,0,1,125,1,2.6,0,0,3)
#   input_data = (age,sex,cp,norm_bps,norm_chol,fbs,norm_thalach,exang)


#   logistic_regression = logisticRegression(input_data)
#   svm = SVM(input_data)
#   dt = decisionTree(input_data)
#   result = (logistic_regression + svm + dt) > 1
#   if result:
#     result = 1
#   else:
#     result = 0
#   return result

# filename = "hhd_after.csv"

# def asd():
#   with open(filename, 'r') as csvfile:
#     csvreader = csv.reader(csvfile)
#     next(csvreader)
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for row in csvreader:
#       # print(row[:-1])
#       age, sex, cp, trestbps, chol, fbs, thalach, exang = row[:-1]
      
#       expected_result = float(row[-1])
#       actual_result = accuracy(age, sex, cp, trestbps, chol, fbs, thalach, exang)

#       if expected_result == 1 and actual_result == 1:
#         TP += 1
#       elif expected_result ==0 and actual_result == 0:
#         TN += 1
#       elif expected_result == 1 and actual_result == 0:
#         FN += 1
#       elif expected_result == 0 and actual_result == 1:
#         FP +=1

#     print(TP, TN, FN, FP)


@app.route('/', methods=['POST'])
def result():
  data = request.get_json()

  age = data['age']
  sex = data['sex']
  cp = data['cp']
  trestbps = data['trestbps']
  chol = data['chol']
  fbs = data['fbs']
  thalach = data['thalach']
  exang = data['exang']
  # oldpeak = data['oldpeak']
  # restecg = data['restecg']
  # slope = data['slope']
  # thal = data['thal']
  # ca = data['ca']

  norm_bps = normalize(trestbps, MAX_TRESTBPS, MIN_TRESTBPS)
  norm_chol = normalize(chol, MAX_CHOL, MIN_CHOL)
  norm_thalach = normalize(thalach, MAX_THALACH, MIN_THALACH)

  # input_data = (60,1,0,145,174,0,1,125,1,2.6,0,0,3)
  input_data = (age,sex,cp,norm_bps,norm_chol,fbs,norm_thalach,exang)


  logistic_regression = logisticRegression(input_data)
  svm = SVM(input_data)
  dt = decisionTree(input_data)
  final_result = (logistic_regression + svm + dt) > 1

  return jsonify({'logistic_regression': str(logistic_regression), 'svm': str(svm), 'dt': str(dt), 'result': str(final_result)})

if __name__ == '__main__':
  # asd()
  app.run(port=8000, debug=True)
