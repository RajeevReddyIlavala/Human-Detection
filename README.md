# Human-Detection

INTRODUCTION:

Implemented a human detection algorithm by using histogram of Oriented Gradients(HOG) features and a linear support vector machine(SVM) classifier.Goal is to generate bounding boxes for each human in static images. Entire model is implemented with the help of NumPy and scikit-learn libraries. In order to train the model, the data is taken from Inria's person dataset, which contains 2,416 positive 64 × 128 images.The negative images are also from same dataset, for each non-pedestrian image, 10 random windows of 64 x 128 pixels were extracted for training, giving a total of 12,180  negative images. This trained model was then used to test the detection accuracy on test images.

TECHNICAL DETAILS:

Using the approach described by Dalal and Triggs, the HOG features are extracted from both positive and negative training images. According to this approach each image is divided in to overlapping blocks and each block is further divided in to cells which will be used to extract one-dimensional histogram of oriented edge magnitudes for entire pixels of a cell. The block sizes can be 2 x 2, 4 x 4 or 8 x 8 and the cell size can 8 x 8 and 16  x 16. All the different combinations were tried to get best results.

The feature vectors extracted from the images were used to train linear SVM classifier. After training 
