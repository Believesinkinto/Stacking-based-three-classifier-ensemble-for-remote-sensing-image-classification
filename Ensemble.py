import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from itertools import cycle
from math import pi
import random
def Process_SVM(X_train,y_train,X_test,y_test,svm_clf):

    svm_clf.fit(X_train, y_train)
    svm_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, svm_pred)
    report = classification_report(y_test, svm_pred)
    print("SVM Classification Report:\n", report)
    return accuracy, svm_pred

def Process_RF(X_train,y_train,X_test,y_test,rf_clf):

    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, rf_pred)
    report = classification_report(y_test, rf_pred)
    print("RF Classification Report:\n", report)
    return accuracy, rf_pred


def Process_logistic(X_train, y_train, X_test, y_test, logistic_clf):
    
    logistic_clf.fit(X_train, y_train)
    logistic_pred = logistic_clf.predict(X_test)
    accuracy = accuracy_score(y_test, logistic_pred)
    report = classification_report(y_test, logistic_pred)
    print("Logistic Regression Classification Report:\n", report)
    return accuracy, logistic_pred

# 计算混淆矩阵并绘图
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()



def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return [accuracy, precision, recall, f1]

# 绘制雷达图
def plot_radar_chart(pred_svm,pred_rf,pred_logistic,weighted_pred,voting_pred, pred_stacking,y_test):

    metrics_df = pd.DataFrame({'SVM': get_metrics(y_test, pred_svm),
                                'Random Forest': get_metrics(y_test, pred_rf),
                                'Logistic Regression': get_metrics(y_test, pred_logistic),
                                'Soft Ensemble': get_metrics(y_test, weighted_pred),
                                'Hard Ensemble': get_metrics(y_test, voting_pred),
                                'Stacking Ensemble': get_metrics(y_test, pred_stacking)}, index=['Accuracy', 'Precision', 'Recall', 'F1'])
    
    metrics_df = metrics_df.T
    categories = list(metrics_df.columns)
    N = len(categories)
    metrics_df.to_csv('metrics_df2.csv')
    # What will be the angle of each axis in the plot? (we divide the plot / number of variables)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each individual model
    for i in range(metrics_df.shape[0]):
        values = metrics_df.iloc[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=metrics_df.index[i])
        ax.fill(angles, values, alpha=0.25)

    # Add the attribute labels to the plot
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.6, 0.7, 0.8, 0.9,1.0], ['0.6', '0.7', '0.8', '0.9','1.0'], color="grey", size=7)
    plt.ylim(0.5, 1)

    # Add a title
    plt.title('Comparison of Model Performance Metrics', size=20, color='black', y=1.1)

    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.show()
    print('over')

def plotResult(pred_svm,pred_rf,pred_logistic,weighted_pred,voting_pred,pred_stacking,y_test):
    plot_confusion_matrix(y_test, pred_svm, 'SVM')
    plot_confusion_matrix(y_test, pred_rf, 'Random Forest')
    plot_confusion_matrix(y_test, pred_logistic, 'Logistic')
    plot_confusion_matrix(y_test, weighted_pred, 'Soft Ensemble')
    plot_confusion_matrix(y_test, voting_pred, 'Hard Ensemble')
    plot_confusion_matrix(y_test, pred_stacking, 'Stacking Ensemble')


    plot_radar_chart(pred_svm,pred_rf,pred_logistic,weighted_pred,voting_pred, pred_stacking,y_test)




def main():

    train_file_path = r'train.csv'
    data = pd.read_csv(train_file_path)

    X_train = data.iloc[:, :-1]
    y_train = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)  

    sorted_bands =  ['band 6', 'band 3', 'band 4', 'band 1', 'band 2', 'band 5']
    band = sorted_bands[0:4]
    
    X_train_selected = X_train[band]
    X_test_selected = X_test[band]

    scaler = StandardScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_test_selected = scaler.transform(X_test_selected)

    svm_params = {'C': 8.0, 'kernel': 'rbf', 'gamma': 8.0, 'probability': True}
    svm_clf = svm.SVC(**svm_params)
    rf_params = {'n_estimators': 200, 'min_samples_split':2, 'min_samples_leaf':1, 'max_depth': 30}
    rf_clf = RandomForestClassifier(**rf_params)
    logistic_params = {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'penalty': 'l2', 'multi_class': 'auto'}  
    logistic_clf = LogisticRegression(**logistic_params)

    accuracy_svm, pred_svm = Process_SVM(X_train_selected,y_train,X_test_selected,y_test,svm_clf)
    accuracy_rf, pred_rf = Process_RF(X_train_selected,y_train,X_test_selected,y_test,rf_clf)
    accuracy_logistic, pred_logistic = Process_logistic(X_train_selected, y_train, X_test_selected, y_test, logistic_clf)

    
    # 元学习器
    # 基础模型
    level0 = [
        ('lr', logistic_clf),
        ('svm', svm_clf),
        ('rf', rf_clf)
    ]
    level1 = LogisticRegression(max_iter=2000, C=1.0)
    # 定义Stacking模型
    stacking_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    stacking_model.fit(X_train, y_train)
    pred_stacking = stacking_model.predict(X_test)
    
    print("Stacking Classification Report:\n", classification_report(y_test, pred_stacking))
    weighted_clf = VotingClassifier(estimators=[('svm', svm_clf),('rf', rf_clf),('logistic', logistic_clf)], voting='soft')
    weighted_clf.fit(X_train_selected, y_train)
    weighted_pred = weighted_clf.predict(X_test_selected)
    print("Weighted Soft Voting Classification Report:\n", classification_report(y_test, weighted_pred))
    # 众数投票
    voting_clf = VotingClassifier(estimators=[('svm', svm_clf),('rf', rf_clf),('logistic', logistic_clf)], voting='hard')
    voting_clf.fit(X_train_selected, y_train)
    voting_pred = voting_clf.predict(X_test_selected)
    print("Weighted Hard Voting Classification Report:\n", classification_report(y_test, voting_pred))

    plotResult(pred_svm,pred_rf,pred_logistic,weighted_pred,voting_pred,pred_stacking,y_test)
    
    



if __name__ == "__main__":

    start_time = datetime.datetime.now()
    print("start...%s" % start_time)

    main()

    end_time = datetime.datetime.now()
    sub_time_days = (end_time - start_time).days
    sub_time_minutes = (end_time - start_time).seconds / 60.0
    print("The program is last %s days , %s minutes" % (sub_time_days, sub_time_minutes))
