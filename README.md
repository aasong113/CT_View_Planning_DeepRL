# CT_View_Planning_DeepRL
Automatic View Planning for Computed Tomography using Deep Reinforcement Learning. 

This method of view planning is adapted from OpenAI gyms 2D Maze. We have created a custom environment that takes a slice of a 3D CT image, and uses Reinforcement Learning to center some anatomical feature of interest into the field of view which is represented as a red square. In our specific example we are locating the sacram of the pelvic anatomy. However, with hyper parameter tuning, we can use other anatomical slices as well. Please refer to the CT_View_Planning_DRL.pdf for all the details, or feel free to contact us. 


If you would like to follow from the beginning, you can see that we started as gym-baby, then grew up into a gym child, then entered the gym-teen years. Much more updates as we become a gym-adult. 


### Getting started 

Go to folder gym-child_thresholded_NN

### Traditional Q-Learning: 

Run the python notebook: Q_Learning.ipynb

You can choose from either a thresholded 9x9 grid intensity based reward function environment, or a gray scale distance based reward function environment. 

https://user-images.githubusercontent.com/54114352/169115926-36d0d335-324c-4656-8e5d-c0aec9f9d88a.mp4


##### Intensity Based Reward Function
This is in the gym child environment. The agent learns to go to the FOV with the largest intensity. In a bone thresholded image, we assume that the center of the sacrum, in pelvic anatomy will have the highest intensity FOV. 

##### Distance Based Reward Function
This reward function is based on a given target position, which means it is a supervised method. The agent learns a path that minimizes the distance between the current position and target position. For this function a target (x,y) position must be inputted to the environment. One notable thing is that when training on different target FOV, the agent will learn the anatomy, such that it will traverse the pelvic anatomy from the illium to the sacrum. This means that it not only takes into account the a path that minimizes the number of steps to get to the target position, but also it learns that the pelvic anatomy is where we are interested and therefore, we must go along this structure to find our target FOV. 

### Deep Q-Learning: 

This uses Deep Learning to approximate Q-learning. You can choose from either a thresholded 9x9 grid intensity based reward function environment, or a gray scale distance based reward function environment. 

#### Fully Connected DQN: 

Flattens the rendered environment image as input into three fully connected layers with ReLU activation. The output is a the number of actions which are up, down, left, right. 

#### Convolutional DQN:

Takes the rendered environment image as input into 3 fully convolutional filters with 2D batchnorm, then outputs to a fully connected layer with output as the number of actions which are up, down, left, right. In the report, this method worked the best for both environments. 
 
