# Human-Detection

INTRODUCTION:
Implemented a human detection algorithm by using histogram of Oriented Gradients(HOG) features and a linear support vector machine(SVM) classifier.Goal is to generate bounding boxes for each human in static images. Entire model is implemented with the help of NumPy and scikit-learn libraries. In order to train the model, the data is taken from Inria's person dataset, which contains 2,416 positive 64 × 128 images.The negative images are also from same dataset, for each non-pedestrian image, 10 random windows of 64 x 128 pixels were extracted for training, giving a total of 12,180  negative images. This trained model was then used to test the detection accuracy on test images.

