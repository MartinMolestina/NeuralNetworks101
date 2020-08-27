import tensorflow as tf
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict(x_test)  # Start predicting using our model
print(np.argmax(predictions[0]))
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()