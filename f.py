

import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt

# Load the California Housing dataset
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Convert the data to a Pandas DataFrame for easier manipulation
columns = california_housing.feature_names
df = pd.DataFrame(data=X, columns=columns)
df['Target'] = y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)


model = Sequential([
    Dense(8, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)


root = tk.Tk()
root.title("AI Regression GUI")

# Label for selecting the feature
feature_label = ttk.Label(root, text="Select Feature:")
feature_label.grid(row=0, column=0, padx=10, pady=5)


selected_feature = tk.StringVar()
feature_dropdown = ttk.Combobox(root, textvariable=selected_feature, values=columns)
feature_dropdown.grid(row=0, column=1, padx=10, pady=5)

plot_button = ttk.Button(root, text="Plot", command=lambda: plot_scatter())
plot_button.grid(row=0, column=2, padx=10, pady=5)

# Canvas for displaying the plot
canvas = tk.Canvas(root, width=600, height=400)
canvas.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

def plot_scatter():
    selected_column = selected_feature.get()
    if selected_column:
        # Scatter plot for the selected feature against the target
        plt.figure(figsize=(8, 5))
        plt.scatter(df[selected_column], df['Target'], alpha=0.5)
        plt.title(f'Scatter Plot of {selected_column} vs Target')
        plt.xlabel(selected_column)
        plt.ylabel('Target')
        plt.grid(True)
        canvas.draw()
    else:
        print("Please select a feature.")


root.mainloop()
