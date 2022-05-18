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

### Deep Q-Learning: 



#### Fully Connected DQN: 

#### Convolutional DQN:
 
