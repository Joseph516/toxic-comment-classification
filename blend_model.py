import pandas as pd
import numpy as np

cnn_gru = pd.read_csv("words_vector/submission.csv")
lstm_cnn= pd.read_csv("words_vector/submission.csv")
lr = pd.read_csv("words_vector/submission.csv")

b1 = cnn_gru.copy()
col = svm.columns

col = col.tolist()
col.remove('id')
print('Blending modle!!!')

# blend models.
# calulation final score acorrding to the 3 models scores.
score_0 = 0.977 
score_1 = 0.987
score_2 = 0.988 

for i in col:
    b1[i] = (score_0 * cnn_gru[i]  + score_1 * lstm_cnn[i] + score_2 * lr[i] ) / (score_0 + score_1 + score_2)

# submission files output 
b1.to_csv('blend_submision.csv', index = False)
