import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 设置保存路径
save_path = "E:\\Term_spread\\"

# 确保目录存在
os.makedirs(save_path, exist_ok=True)

# 读取数据
file_path = "E:\\Term_spread\\SCFP2009panel.xlsx"
df = pd.read_excel(file_path)

# 选择相关列
risky_assets_cols = ["STOCKS07", "BOND07", "EQUITY07"]
risk_free_assets_cols = ["CDS07", "CASHLI07"]

# 计算总资产
df["Risky_Assets"] = df[risky_assets_cols].sum(axis=1)
df["Risk_Free_Assets"] = df[risk_free_assets_cols].sum(axis=1)

# 计算风险容忍度
df["Risk_Tolerance"] = df["Risky_Assets"] / (df["Risky_Assets"] + df["Risk_Free_Assets"])

# 删除无法计算风险容忍度的行
df = df[df["Risk_Tolerance"].notna()]

# 归一化风险容忍度（基于 S&P 500 2007 和 2009）
SP500_2007 = 1478
SP500_2009 = 948
df["Risk_Tolerance_Normalized"] = df["Risk_Tolerance"] * (SP500_2007 / SP500_2009)

# 📌 1. 绘制投资者风险容忍度分布图
plt.figure(figsize=(10, 5))
sns.histplot(df["Risk_Tolerance_Normalized"], bins=50, kde=True, color="blue", label="Risk Tolerance")
plt.xlabel("Risk Tolerance (Normalized)")
plt.ylabel("Frequency")
plt.title("Distribution of Investor Risk Tolerance")
plt.legend()
plt.savefig(os.path.join(save_path, "risk_tolerance_distribution.png"))  # 保存图像
plt.close()

# 📌 2. 计算并绘制相关性矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(df[["Risk_Tolerance", "Risky_Assets", "Risk_Free_Assets"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig(os.path.join(save_path, "correlation_matrix.png"))  # 保存图像
plt.close()

# 📌 3. 线性回归建模
features = ["AGE07", "EDUC07", "WAGEINCPCT", "LEVERAGEPCT"]
X = df[features]
y = df["Risk_Tolerance_Normalized"]

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
r2 = r2_score(y_test, y_pred)
print(f"Regression Model R² Score: {r2:.4f}")

# 📌 4. 保存数据和模型
df.to_csv(os.path.join(save_path, "Processed_SCData.csv"), index=False)
print(f"Processed data saved as: {save_path}Processed_SCData.csv")

import pickle
with open(os.path.join(save_path, "risk_tolerance_model.pkl"), "wb") as file:
    pickle.dump(model, file)
print(f"Regression model saved as: {save_path}risk_tolerance_model.pkl")
