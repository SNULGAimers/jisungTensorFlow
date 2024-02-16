import pandas as pd 
import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np

titanic = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
print(titanic.head())
print(titanic.shape)

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

inputs = {}

for name, column in titanic_features.items():
    dtype = column.dtype 
    if dtype ==object:
        dtype = tf.string
    else:
        dtype = tf.float32 
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

numeric_inputs = {name:input for name, input in inputs.items() if input.dtype==tf.float32}

x=layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_input = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue 
    lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_input.append(x)

preprocessed_input_cat = layers.Concatenate()(preprocessed_input)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_input_cat) #계산은 이 모델을 실행해야 이루어짐

tf.keras.utils.plot_model(model=titanic_preprocessing, rankdir='LR', dpi=72, show_shapes=True)

titanic_feature_dict = {name: np.array(value) for name, value in titanic_features.items()}

x = tf.data.Dataset.from_tensor_slices((titanic_features, titanic_labels)).shuffle(507).batch(50)
y = tf.data.Dataset.from_tensor_slices((titanic_features[507:], titanic_labels[507:])).batch(50)



def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1)
    ])
    preprocessed_input = preprocessing_head(inputs) # 전처리된 모델을 얻음
    print(preprocessed_input.shape)
    result = body(preprocessed_input) 
    model = tf.keras.Model(inputs, result) # 실행 모델

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.legacy.Adam(),
                  metrics=['accuracy']) # 마찬가지로 계산은 여기서 이루어짐
    return model 


titanic_model = titanic_model(titanic_preprocessing, inputs)


titanic_model.fit(x=titanic_feature_dict, y=titanic_labels, epochs=10)

