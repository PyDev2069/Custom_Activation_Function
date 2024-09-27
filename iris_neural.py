import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score
def custom_activation(x):    
    return 0.1*x/2*(x*(math.e **x))  
def output_activation(x):
    return (math.e**x)/1+(math.e**x)
iris = load_iris()
X = iris.data  
y = iris.target  
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(X_train.shape[1],), activation=custom_activation),  
    tf.keras.layers.Dense(3, activation=output_activation)  
])
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.save('api.h5')
history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
final_train_accuracy = history.history['accuracy'][-1]  
print(f"Final Training Accuracy: {final_train_accuracy * 100:.2f}%")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)  
print(f"Final Test Accuracy on the whole test set: {test_acc * 100:.2f}%")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted classes:", predicted_classes)
true_classes = np.argmax(y_test, axis=1)
f1 = f1_score(true_classes, predicted_classes, average='weighted')  
print(f'Weighted F1 Score: {f1:.2f}')

for layer in model.layers:
    weights, biases = layer.get_weights()
    print("Weights:", weights)
    print("Biases:", biases)

import matplotlib.pyplot as plt
history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0, validation_split=0.2)


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


