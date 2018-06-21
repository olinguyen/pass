import numpy as np
from keras.utils import plot_model
from models.nn_models import *

embedding_matrix = np.load('./models/embedding_matrix.npz.npy')
lstm = get_lstm_model(embedding_matrix)
cnn = get_cnn_model(embedding_matrix)

plot_model(cnn, './results/cnn.png', show_shapes=True)
plot_model(lstm, './results/lstm.png', show_shapes=True)

