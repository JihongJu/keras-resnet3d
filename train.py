import numpy as np
from resnet3d import Resnet3DBuilder

# pseudo volumetric data
X_train = np.random.rand(10, 64, 64, 32, 1)
labels = np.random.randint(0, 2, size=[10])
y_train = np.eye(2)[labels]

# train
model = Resnet3DBuilder.build_resnet_18((64, 64, 32, 1), 2)
model.compile(loss="categorical_crossentropy", optimizer="sgd")
model.fit(X_train, y_train, batch_size=10)
