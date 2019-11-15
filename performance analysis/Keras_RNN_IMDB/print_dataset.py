from keras.datasets import imdb

max_features = 20000

# (x_train, y_train), (x_test, y_test) = imdb.load_data()
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# print(x_train[:10])
for x in x_train[:10]:
    print(len(x))

print(y_train[:10])
