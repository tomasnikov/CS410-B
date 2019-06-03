import sys
import json
import numpy as np
from matplotlib import pyplot as plt

print(sys.argv)
file = sys.argv[1]
adversarial = file.replace("datasets", "adversarials")

orig = open(file, "r")
orig_data = json.loads(orig.read())
img = np.resize(orig_data['input'], (28,28))
img_label = orig_data['label']
print(img_label)

adv = open(adversarial, "r")
adv_data = json.loads(adv.read())
adv_img = np.resize(adv_data['input'], (28,28))
adv_label = adv_data['label']
print(adv_label)

fig1 = plt.figure()
plt.imshow(img, cmap = 'gray')
plt.title("Original Image, Label: %s" % img_label)
fig2 = plt.figure()
plt.imshow(adv_img, cmap = 'gray')
plt.title("Adversarial Image, Predicted Label: %s" % adv_label)
plt.show()