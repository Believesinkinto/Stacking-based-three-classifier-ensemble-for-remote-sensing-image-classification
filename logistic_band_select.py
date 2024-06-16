#logistic回归多分类问题的python代码，用于解决6波段遥感影像地物分类问题
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tools import add_constant
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
import os
import datetime 
from sklearn.preprocessing import StandardScaler   
  
def open_txt_film(filepath):
    # open the film
    if os.path.exists(filepath):
        with open(filepath, mode='r') as f:
            train_data_str = np.loadtxt(f, delimiter=' ')
            print('训练（以及测试）数据的行列数为{}'.format(train_data_str.shape))
            return train_data_str
 # read the train data from txt film

# 前向选择代码
def forward_stepwise_selection(X, y):
    initial_features = []
    remaining_features = list(X.columns)
    best_features = []
    while remaining_features:
        scores_with_candidates = []
        best_score = -np.inf
        best_feature = None
        for feature in remaining_features:
            features_to_test = initial_features + [feature]
            X_train_const = add_constant(X[features_to_test])
            logit_model = sm.MNLogit(y, X_train_const).fit(disp=0)
            score = logit_model.llf
            if score > best_score:
                best_score = score
                best_feature = feature


        if best_feature is not None:
            initial_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_features.append(best_feature)
            print(f'Added feature: {best_feature}, Score: {best_score}')
        else:
            break
    
    return best_features



def main():

    train_file_path = r'train.csv'


    data = pd.read_csv(train_file_path)
    train_data_x = data.iloc[:, :-1]
    train_data_y = data.iloc[:, -1]
    
    # 数据预处理：分割数据集为训练集和测试集  
    X_train, X_test, y_train, y_test = train_test_split(train_data_x, train_data_y, test_size=0.3, random_state=42)  


    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

  # 将标准化后的数据转换回DataFrame格式，并保留索引
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=train_data_x.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=train_data_x.columns, index=X_test.index)
    # 添加常数项
    X_train_const = add_constant(X_train_scaled)
    # 创建并训练模型
    logit_model = sm.MNLogit(y_train, X_train_const).fit()
    # 打印摘要，显示系数的统计检验结果
    print(logit_model.summary())

        # 预测
    # 捕获 summary() 方法的输出  
    summary_text = logit_model.summary().as_text()  # 注意：这取决于你使用的库，可能需要调整  
    
    # 将输出写入到文件中  
    with open('logit_model_summary.txt', 'w', encoding='utf-8') as f:  
        f.write(summary_text)  

    y_pred = logit_model.predict(add_constant(X_test_scaled))
    y_pred_class = y_pred.idxmax(axis=1)+1  # 获取预测的类别

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy: {accuracy}")
    # 执行逐步选择
    selected_features = forward_stepwise_selection(X_train_scaled, y_train)
    print(f'Selected features: {selected_features}')




if __name__ == "__main__":
    # remember the beginning time of the program
    start_time = datetime.datetime.now()
    print("start...%s" % start_time)

    main()

    # record the running time of program with the unit of minutes
    end_time = datetime.datetime.now()
    sub_time_days = (end_time - start_time).days
    sub_time_minutes = (end_time - start_time).seconds / 60.0
    print("The program is last %s days , %s minutes" % (sub_time_days, sub_time_minutes))
