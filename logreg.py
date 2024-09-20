import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.title('Логистическая регрессия')

uploaded_file = st.file_uploader("__Выберите файл CSV__", type="csv")

if uploaded_file is not None:
    train = pd.read_csv(uploaded_file)

    st.write('### Данные:')
    st.write(train)

    X = train.drop('Personal.Loan', axis=1)
    y = train['Personal.Loan']

    from sklearn.preprocessing import StandardScaler
    s_scaler = StandardScaler()
    train[['CCAvg', 'Income']] = s_scaler.fit_transform(train[['CCAvg', 'Income']])

    class LogReg:
        def __init__(self, n_inputs, learning_rate=0.001, n_epochs=1000):
            self.learning_rate = learning_rate
            self.n_inputs = n_inputs # кол-во фичей
            self.n_epochs = n_epochs # кол-во итераций
            self.coef_ = np.random.uniform(-1, 1, size=self.n_inputs) # веса
            self.intercept_ = np.random.uniform(-1, 1, size=1) # w0

        def sigmoid(self, z):
            s = 1 / (1 + np.exp(-z))
            return s
        
        def fit(self, X, y):
            X = np.array(X)
            y = np.array(y)
            
            for _ in range(self.n_epochs):
                # у-предсказанное / сигмоида
                y_pred = self.sigmoid(X@self.coef_ + self.intercept_)
                
                # Градиент
                dw = -X * (y - y_pred).reshape(-1, 1) # веса w1,2,..,n
                dw_0 = -(y - y_pred).reshape(-1, 1) # w0

                # Обновление весов, смещения (w_0)
                self.coef_ = self.coef_ - self.learning_rate * dw.mean(axis=0)
                self.intercept_ = self.intercept_ - self.learning_rate * dw_0.mean()
                # Наглядности ради
                print(f'Значения весов - {self.coef_}')
                print(f'Значение смещения - {self.intercept_}')

        def predict(self, X):
            X = np.array(X)
            y_pred = self.sigmoid(self.intercept_ + X@self.coef_)
            return y_pred   

    lr = LogReg(2)
    lr.fit(train[['CCAvg', 'Income']], train['Personal.Loan'])

    # Результат регрессии
    st.write('### Результат регрессии:')
    coef_dict = {column: coef for column, coef in zip(X.columns, lr.coef_)}
    st.write(coef_dict)
    
    # Выбор фич

    st.write('### Выберите фичи для построения точечного графика:')
    x_feature = st.selectbox('X фича', X.columns)
    y_feature = st.selectbox('Y фича', X.columns)

    # TRAIN SCATTER PLOT

    X_train = train[['CCAvg', 'Income']]
    y_train = train['Personal.Loan']

    # Предсказание значений
    y_pred = lr.predict(train[['CCAvg', 'Income']])

    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(train[x_feature], train[y_feature], c=y, cmap='coolwarm')
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title('Logistic Regression')

    #Плоскость
    x = np.linspace(train[y_feature].min(), train[x_feature].max(), 1000)
    ax.plot(x, (-lr.coef_[0] * x - lr.intercept_ / lr.coef_[1]), color='g')
    
    st.pyplot(fig)
    


else:
    st.write('Загрузите файл CSV')

