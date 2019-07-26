# Maze World - Assignment 2
Assignment code for course ECE 493 T25 at the University of Waterloo in Spring 2019.
(*Code designed and created by Sriram Ganapathi Subramanian and Mark Crowley, 2019*)

**Due Date:** July 30 11:59pm submitted as PDF and code to LEARN dropbox.

**Collaboration:** You can discuss solutions and help to work out the code. But each person *must do their own work*. All code and writing will be cross-checked against each other and against internet databases for cheating. 

Updates to code which will be useful for all or bugs in the provided code will be updated on gitlab and announced.

## Domain Description - GridWorld
The domain consists of a 10x10 grid of cells. The agent being controlled is represented as a red square. The goal is a yellow oval and you receive a reward of 1 for reaching it, this ends and resets the episode.
Blue squares are **pits** which yield a penalty of -10 and end the episode. 
Black squares are **walls** which cannot be passed through. If the agent tries to walk into a wall they will remain in their current position and receive a penalty of -.3.
Their are **three tasks** defined in `run_main.py` which can be commented out to try each. They include a combination of pillars, rooms, pits and obstacles. The aim is to learn a policy that maximizes expected reward and reaches the goal as quickly as possible.

# <img src="task1.png" width="300"/><img src="task2.png" width="300"/><img src="task3.png" width="300"/>

## Assignment Requirements

This assignment will have a written component and a programming component.
Clone the mazeworld environment locally and run the code looking at the implemtation of the sample algorithm.
Your task is to implement three other algortihms on this domain.
- **(20%)** Implement SARSA
- **(20%)** Implement QLearning
- **(20%)** At least one other algorithm of your choice or own design. 
Suggestions to try:
    - Policy Iteration (easy)
    - Expected SARSA (less easy)
    - Double Q-Learning (less easy)
    - n-step TD or TD(Lambda) with eligibility traces (harder)
    - Policy Gradients (harderer)
- **(10%) bonus** Implement four algorithms in total (you can do more but we'll only look at four, you need to tell us which).
- **(40%)** Report : Write a short report on the problem and the results of your three algorithms. The report should be submited on LEARN as a pdf. 
    - Describing each algorithm you used, define the states, actions, dynamics. Define the mathematical formulation of your algorithm, show the Bellman updates for you use.
    - Some quantitative analysis of the results, a default plot for comparing all algorithms is given. You can do more than that.
    - Some qualitative analysis of why one algorithm works well in each case, what you noticed along the way.


### Evaluation
You will also submit your code to LEARN and grading will be carried out using a combination of automated and manual grading.
Your algorithms should follow the pattern of the `RL_brain.py` and `RL_brainsample_PI.py` files.
We will look at your definition and implmentation which should match the description in the document.
We will also automatically run your code on the given domain on the three tasks define in `run_main.py` as well as other maps you have not seen in order to evaluate it. 
Part of your grade will come from the overall performance of your algorithm on each domain.
So make sure your code runs with the given unmodified `run_main` and `maze_end` code if we import your class names.


### Code Suggestions
- When the number of episodes ends a plot is displayed of the algorithm performance. If multiple algorithms are run at once then they will be all plotted together for comparison. You may modify the plotting code and add any other analysis you need, this is only a starting point.
- there are a number of parameters defined in `run_main` that can be used to speed up the simulations. Once you have debugged an algorithm and see it is running you can alter the `sim_speed`, `\*EveryNth` variables to alter the speed of each step and how often data is printed or updated visually to speed up training. 
- For the default algorithms we have implmented on these domains it seems to take at least 1500 episodes to converge, so don't read too much into how it looks after a few hundred.

<img src="plot.png" width="400"/><img src="plotzoom.png" width="400"/>
