import tensorflow as tf

mnist = tf.keras.datasets.mnist;
[xTrain, yTrain], [xTest, yTest] = mnist.load_data()

xTrain = tf.keras.utils.normalize(xTrain, axis=1);
xTest = tf.keras.utils.normalize(xTest, axis=1);

model = tf.keras.models.load_model('HandWrittenModel.h5');

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']);

model.fit(xTrain, yTrain, epochs=100);

model.save('HandWrittenModel.h5');