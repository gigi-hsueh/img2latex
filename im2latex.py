#Gigi Hsueh
#HW6
#image processing and tokenizing and trainings are all successful
#this program is to be run with 'python im2_latex.py'
#additional arguments can be added, but predictions are not ready to run yet 
#image folder paths may need to be changed to run accordingly

import sys
from skimage import io
import numpy as np
import csv
import re
from scipy.misc import imresize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from preprocess_formulas import tokenize_formula, remove_invisible, normalize_formula

#images folder
image_train= "./im2latex_train.lst"
image_path="./formula_images/"
formulas_path="./im2latex_formulas.lst"


class LatexTranslator:

    def __init__(self, learning_rate=0.001, 
            image_size=50, split=0.2):

        #initializing things
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.split = split
        self.im_train = {}
        self.image_data = []
        self.token_list= {}
        self.targets_vector = []
        self.target = []
        self.labels = []
        self.index = []
        self.i = 0
        self.maxlen = 10

    def extract_label(self, image_name):
        label_name = image_name.split('/')[2]
        label_name = label_name.split('.')[0]
        
        return label_name

    def image_processing(self, image):
        #takes each image, crop, resize, reshape, and then add to an array
        label = self.extract_label(image)
        img = io.imread(image, as_gray=True)
        img = img[300:1550, 330:1580]
        img = imresize(img, (self.image_size, self.image_size)) 
        img = img.reshape([self.image_size, self.image_size, 1])           
        self.image_data.append(img)
        self.labels.append(np.array(label))

    def formula_processing(self, formulas):
        #only taking the formulas that are smaller than length of 150
        #using the tokenizer, it removes invisible, noralize the formula, and then tokenize the formula
        #then for special token, they're saved to a dictionary assigning an integer to each token
        #Also, each formulas are turned into integer values for easier use of target
        token_for = []
        formula = remove_invisible(formulas)
        formula = normalize_formula(formula)
        tokenized = tokenize_formula(formula)
        for token in tokenized:
            if token not in self.token_list:
                self.token_list[token] = self.i 
                self.i = self.i + 1
            token_for.append(self.token_list[token])

        if (len(token_for) != self.maxlen):
            for i in (range(self.maxlen-len(token_for))):
                token_for.append(0)

        return token_for


    def open_formula(self, index):
        #opens the formula file line by line, for each line, process it
        f = open(formulas_path, "r", errors='ignore')
        formula = f.readlines()
        f.close()

        if (len(formula[index]) <= 200):
            self.index.append(index)
            if (len(formula[index]) > self.maxlen):
                self.maxlen = len(formula[index])
            t = self.formula_processing(formula[index])
            self.target.append(t)


    def values_to_targets(self, values):
        # #integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        #binary encode 
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        self.targets_vector = onehot_encoded


    def predict_image(self, image):
        #this is for after the trained_model, it makes a prediction about the model and output an array of probabilities
        results = self.model.predict([img])[0]
        most_probable = max(results)
        results = list(results)
        most_probable_index = results.index(most_probable)
        return results

    def load_model(self, model_file):
        #loading the model saved 
        model = self.build_cnn()
        model.load(model_file)
        self.model = model


    def build_cnn(self):
        #CNN
        convnet = input_data(shape=[None, self.image_size, self.image_size, 1], name='input')
        
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
      
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 200, activation='softmax')
        convnet = regression(convnet, optimizer='adam',
                             learning_rate=self.learning_rate,
                             loss='binary_crossentropy',
                             name='targets')

        model = tflearn.DNN(convnet,
                            tensorboard_dir='log',
                            tensorboard_verbose=0
        )

        return model


    def train_model(self, n_epochs=10, batch_size=2):
        #input is our images
        #output is the latex formulas
        X = self.image_data
        y = self.target


        model = self.build_cnn()

        model.fit(X, y,
                      n_epoch=10,
                      validation_set=0.1,
                      snapshot_step = 500,
                      show_metric=True,
            batch_size=batch_size)

        model.save('network.tflearn')
        


if __name__ == '__main__':

    # n_epochs = sys.argv[1]
    # m_model = sys.argv[2]
    # test_images_list = sys.argv[3]

    lt = LatexTranslator(learning_rate=0.001,
                image_size=50, split=0.2)
    numtrain = 100
    train = {}

    with open(image_train, "r") as data:
            for line in data.readlines()[:numtrain]:
                parts = line.split(' ')
                train[parts[0]] = parts[1] 
                lt.open_formula(int(parts[0]))
                for index in lt.index:
                    if (int(parts[0])==index):
                        lt.image_processing('./formula_images/'+parts[1]+'.png')


    lt.train_model(
        n_epochs=10,
        batch_size=5)

    # lt.load_model(m_model)

    # with open(test_images_list, "r") as data:
    #         for line in data.readlines()[:numtrain]:
    #             parts = line.split(' ')
    #             train[parts[0]] = parts[1] 
    #             lt.open_formula(int(parts[0]))
    #             for index in lt.index:
    #                 if (int(parts[0])==index):
    #                     lt.image_processing('./formula_images/'+parts[1]+'.png')


    # X = lt.image_data
    # y = lt.targets_vector

    # results = lt.model.predict(X)
    # #most_probable = max(results)
    # results = list(results)
    # most_probable_index = results.index(most_probable)
