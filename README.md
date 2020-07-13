# Monocular-Depth-Perception
The goal of this project is to develop a machine learning model to generate depth maps of scenes captured using monocular cameras that can match the accuracy of existing LIDAR equipment and state-of-the-art algorithms. 

# Model:
We used the keras resnet50 library to generate our model and used the KITTI dataset to train our model. In order to meet the project requirements we had to preprocess the KITTI images into 4 256x256 images. 

# Application:
After training our model, we generated a weights files in order to load it into our application. Our application using the Kivy module in python and allows the user to generate depth maps using our model.

# Demo:
A short demo of our application in action. 

# Acknowledgements:
I would like to thank my partners Sofian Zalouk and James Meijers.
