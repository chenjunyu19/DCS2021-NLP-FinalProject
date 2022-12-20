import json
import os
import matplotlib.pyplot as plt

import utils

CONFIG = utils.read_config()

with open(os.path.join(CONFIG['dataDir'], 'losses.json'), 'r') as f:
    losses = json.load(f)

data_x = range(len(losses))
data_loss = [i['loss'] for i in losses]
data_val_loss = [i['val_loss'] for i in losses]

plt.plot(data_x, data_loss, label='Train loss')
plt.plot(data_x, data_val_loss, label='Eval loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses.svg')
plt.show()
