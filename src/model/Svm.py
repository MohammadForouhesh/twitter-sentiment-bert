from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
              'C': [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
              'tol': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]}


def svm_grid(X_train, X_test, y_train, y_test):
    svc = SVC(class_weight='balanced', probability=True)
    clf = GridSearchCV(svc, parameters, scoring='roc_auc_ovr')
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    svc = SVC(class_weight='balanced', **clf.best_params_)

    svc.fit(X_train, y_train)
    pred = svc.predict(X_train)

    print(classification_report(y_train, pred))

    pred = svc.predict(X_test)

    print(classification_report(y_test, pred))
    plot_confusion_matrix(svc, X_test, y_test)
    plt.show()