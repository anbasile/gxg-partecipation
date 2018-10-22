"""
Library for the evaluation
"""

from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import classification_report as report
from sklearn.metrics import confusion_matrix

def run(x,y):
    print(accuracy(x,y))
    print(report(x,y))
    print(confusion_matrix(x,y))
    return 0
