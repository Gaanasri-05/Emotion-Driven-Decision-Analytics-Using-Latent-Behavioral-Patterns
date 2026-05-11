# src/evaluation.py

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, labels=["deliberate", "impulsive"]):
    """
    Evaluate the trained model.

    Returns:
        metrics_dict: dictionary with accuracy and report
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.show()
    
    metrics_dict = {"accuracy": accuracy, "report": report, "y_pred": y_pred}
    return metrics_dict
