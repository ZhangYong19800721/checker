import pickle
import DATASET
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_file = open(r"./model/model.pkl", "rb")
model = pickle.load(model_file)
model.to(device)
model_file.close()

testset = DATASET.GCDYW(r"./data/corpus_trainset_part1_digit.cps")  # load the test data
minibatch_size = 20
dataloader = DATASET.LOADER(testset, batch_size=minibatch_size)  # set the minibatch size
minibatch_num = len(dataloader)

error_count = 0
with torch.no_grad():
    for minibatch_id in range(minibatch_num):
        minibatch = dataloader[minibatch_id]
        article = minibatch['article'].to(device)
        label = minibatch['label'].numpy()
        predict = model(article).to('cpu').numpy()
        predict = np.argmax(predict, axis=1)
        error_count += np.sum((predict != label) + 0.0)

error_rate = error_count / (minibatch_size * minibatch_num)
print(f"error_rate = {error_rate}")