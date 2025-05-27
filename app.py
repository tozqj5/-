import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

config = {
    '平焊': ('t85平.csv', 0.001),
    '立焊': ('t85立.csv', 0.001),
    '仰焊': ('t85仰.csv', 0.002),
}

class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_and_predict(method, new_input):
    file_path, lr = config[method]
    df = pd.read_csv(file_path, encoding='gbk')
    X = df[['热输入', '电流', '电压', '速度']].values
    y = np.log1p(df[['t8/5']].values)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    model = MLPRegressor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.FloatTensor(X_train_scaled)
    y_tensor = torch.FloatTensor(y_train_scaled)

    for _ in range(500):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    new_input_scaled = scaler_X.transform([new_input])
    new_tensor = torch.FloatTensor(new_input_scaled)

    model.eval()
    with torch.no_grad():
        predicted_scaled = model(new_tensor)
        predicted_log = scaler_y.inverse_transform(predicted_scaled.numpy())
        predicted_t85 = np.expm1(predicted_log)

    return predicted_t85[0][0]

# Streamlit UI
st.title("t8/5 焊接参数预测")

method = st.selectbox("选择焊接方式", list(config.keys()))
heat_input = st.number_input("热输入 (kJ/cm)", value=20.0)
current = st.number_input("电流 (A)", value=200.0)
voltage = st.number_input("电压 (V)", value=25.0)
speed = st.number_input("速度 (cm/min)", value=30.0)

if st.button("预测"):
    try:
        result = train_and_predict(method, [heat_input, current, voltage, speed])
        st.success(f"{method} 的预测 t8/5 为: {result:.2f} 秒")
    except Exception as e:
        st.error(f"发生错误: {e}")
