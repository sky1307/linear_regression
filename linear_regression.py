from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()

def synthetic_data(w, b, num_examples):
    """Generate y = X w + b + noise. """
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y

true_w = np.array([2, -3,4])
true_b = 4.2
features, labels = synthetic_data(true_w,true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])