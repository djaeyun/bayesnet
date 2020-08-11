## CS 6601 Assignment 3: Bayes Nets

In this assignment, you will work with probabilistic models known as Bayesian networks to efficiently calculate the answer to probability questions concerning discrete random variables.

### Resources

You will find the following resources helpful for this assignment.

*Canvas Videos:*  
[Lecture 5 on Probability](https://gatech.instructure.com/courses/151546/pages/5-probability?module_item_id=806766)  
[Lecture 6 on Bayes Nets](https://gatech.instructure.com/courses/151546/pages/6-bayes-nets?module_item_id=806856)  

*Textbook:*   
Chapter 13: Quantifying Uncertainty  
Chapter 14: Probabilistic Reasoning  

*Others:*   
[Markov Chain Monte Carlo](https://github.gatech.edu/omscs6601/assignment_3/blob/master/resources/LESSON1_Notes_MCMC.pdf)  
[Gibbs Sampling](http://gandalf.psych.umn.edu/users/schrater/schrater_lab/courses/AI2/gibbs.pdf)  
[Metropolis Hastings Sampling - 1](https://github.gatech.edu/omscs6601/assignment_3/blob/master/resources/mh%20sampling.pdf)  
[Metropolis Hastings Sampling - 2](http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf)  

### Setup

1. Clone the project repository from Github

   ```
   git clone https://github.gatech.edu/omscs6601/assignment_3.git
   ```

2. Navigate to `assignment_3/` directory

3. Activate the environment you created during Assignment 0 

    ```
    conda activate ai_env
    ```
    
    In case you used a different environment name, to list of all environments you have on your machine you can run `conda env list`.

4. Run the following command in the command line to install and update the required packages

    ```
    pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install --upgrade -r requirements.txt
    ```

### Submission

Please include all of your own code for submission in `submission.py`.  

**Important: There is a TOTAL submission limit of 5 on Gradescope for this assignment. This means you can submit a maximum of 5 times during the duration of the assignment. Please use your submissions carefully and do not submit until you have thoroughly tested your code locally.**

**If you're at 4 submissions, use your fifth and last submission wisely. The submission marked as ‘Active’ in Gradescope will be the submission counted towards your grade.**

### Restrictions

You are not allowed to use following set of modules from 'pgmpy' Library.

>- pgmpy.sampling.*
>- pgmpy.factor.*
>- pgmpy.estimators.*

## Part 1 Bayesian network tutorial:

_[35 points total]_

To start, design a basic probabilistic model for the following system:

There's a nuclear power plant in which an alarm is supposed to ring when the gauge reading exceeds a fixed threshold. The gauge reading is based on the actual temperature, and for simplicity, we assume that the temperature is represented as either high or normal. However, the alarm is sometimes faulty. The temperature gauge can also fail, with the chance of failing greater when the temperature is high.

You will test your implementation at the end of each section.

### 1a: Casting the net

_[10 points]_

Assume that the following statements about the system are true:
> 1. The temperature gauge reads the correct temperature with 95% probability when it is not faulty and 20% probability when it is faulty. For simplicity, say that the gauge's "true" value corresponds with its "hot" reading and "false" with its "normal" reading, so the gauge would have a 95% chance of returning "true" when the temperature is hot and it is not faulty.
> 2. The alarm is faulty 15% of the time.
> 3. The temperature is hot (call this "true") 20% of the time.
> 4. When the temperature is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% of the time.
> 5. The alarm responds correctly to the gauge 55% of the time when the alarm is faulty, and it responds correctly to the gauge 90% of the time when the alarm is not faulty. For instance, when it is faulty, the alarm sounds 55% of the time that the gauge is "hot" and remains silent 55% of the time that the gauge is "normal."


Use the following name attributes:

>- "alarm"
>- "faulty alarm"
>- "gauge"                   (high = True, normal = False)
>- "faulty gauge"
>- "temperature"             (high = True, normal = False)  

Use the description of the model above to design a Bayesian network for this model. The `pgmpy` package is used to represent nodes and conditional probability arcs connecting nodes. Don't worry about the probabilities for now. Use the functions below to create the net. You will write your code in `submission.py`. 

Fill in the function `make_power_plant_net()`

The following commands will create a BayesNet instance add node with name "alarm":

    BayesNet = BayesianModel()
    BayesNet.add_node("alarm")

You will use `BayesNet.add_edge()` to connect nodes. For example, to connect the alarm and temperature nodes that you've already made (i.e. assuming that temperature affects the alarm probability):

Use function `BayesNet.add_edge(<parent node name>,<child node name>)`
    
    BayesNet.add_edge("temperature","alarm")

After you have implemented `make_power_plant_net()`, you can run the following test in the command line to make sure your network is set up correctly.

```
python probability_tests.py ProbabilityTests.test_network_setup
```

### 1b: Setting the probabilities

_[15 points]_

Now set the conditional probabilities for the necessary variables on the network you just built.

Fill in the function `set_probability()`

Using `pgmpy`'s `factors.discrete.TabularCPD` class: if you wanted to set the distribution for node 'A' with two possible values, where P(A) to 70% true, 30% false, you would invoke the following commands:

    cpd_a = TabularCPD('A', 2, values=[[0.3], [0.7]])

**NOTE: Use index 0 to represent FALSE and index 1 to represent TRUE, or you may run into testing issues.**

If you wanted to set the distribution for P(A|G) to be

|  G  |P(A=true given G)|
| ------ | ----- |
|  T   | 0.75|
|  F   | 0.85| 

you would invoke:

    cpd_ag = TabularCPD('A', 2, values=[[0.15, 0.25], \
                        [ 0.85, 0.75]], evidence=['G'], evidence_card=[2])

Modeling a three-variable relationship is a bit trickier. If you wanted to set the following distribution for P(A|G,T) to be

| G   |  T  |P(A=true given G and T)|
| --- | --- |:----:|
|T|T|0.15|
|T|F|0.6|
|F|T|0.2|
|F|F|0.1|

you would invoke

    cpd_agt = TabularCPD('A', 2, values=[[0.9, 0.8, 0.4, 0.85], \
                        [0.1, 0.2, 0.6, 0.15]], evidence=['G', 'T'], evidence_card=[2, 2])

The key is to remember that first entry represents the probability for P(A==False), and second entry represents P(A==true).

Add Tabular conditional probability distributions to the bayesian model instance by using following command.

    bayes_net.add_cpds(cpd_a, cpd_ag, cpd_agt)


You can check your probability distributions in the command line with

```
python probability_tests.py ProbabilityTests.test_probability_setup
```

### 1c: Probability calculations : Perform inference

_[10 points]_

To finish up, you're going to perform inference on the network to calculate the following probabilities:

> - the marginal probability that the alarm sounds
> - the marginal probability that the gauge shows "hot"
> - the probability that the temperature is actually hot, given that the alarm sounds and the alarm and gauge are both working

You'll fill out the "get_prob" functions to calculate the probabilities:
- `get_alarm_prob()`
- `get_gauge_prob()`
- `get_temperature_prob()`

Here's an example of how to do inference for the marginal probability of the "faulty alarm" node being True (assuming `bayes_net` is your network):

    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['faulty alarm'], joint=False)
    prob = marginal_prob['faulty alarm'].values
  
To compute the conditional probability, set the evidence variables before computing the marginal as seen below (here we're computing P('A' = false | 'B' = true, 'C' = False)):


    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['A'],evidence={'B':1,'C':0}, joint=False)
    prob = conditional_prob['A'].values
    
__NOTE__: `marginal_prob` and `conditional_prob` return two probabilities corresponding to `[False, True]` case. You must index into the correct position in `prob` to obtain the particular probability value you are looking for. 

If you need to sanity-check to make sure you're doing inference correctly, you can run inference on one of the probabilities that we gave you in 1a. For instance, running inference on P(T=true) should return 0.20 (i.e. 20%). However, due to imprecision in some machines it could appear as 0.199xx. You can also calculate the answers by hand to double-check.

## Part 2: Sampling

_[65 points total]_

For the main exercise, consider the following scenario.

There are three frisbee teams who play each other: the Airheads, the Buffoons, and the Clods (A, B and C for short). 
Each match is between two teams, and each team can either win, lose, or draw in a match. Each team has a fixed but 
unknown skill level, represented as an integer from 0 to 3. The outcome of each match is probabilistically proportional to the difference in skill level between the teams.

Sampling is a method for ESTIMATING a probability distribution when it is prohibitively expensive (even for inference!) to completely compute the distribution. 

Here, we want to estimate the outcome of the matches, given prior knowledge of previous matches. Rather than using inference, we will do so by sampling the network using two [Markov Chain Monte Carlo](http://www.statistics.com/papers/LESSON1_Notes_MCMC.pdf) models: Gibbs sampling (2c) and Metropolis-Hastings (2d).

### 2a: Build the network.

_[10 points]_

For the first sub-part, consider a network with 3 teams : the Airheads, the Buffoons, and the Clods (A, B and C for short). 3 total matches are played. 
Build a Bayes Net to represent the three teams and their influences on the match outcomes. 

Fill in the function `get_game_network()`

Assume the following variable conventions:

| variable name | description|
|---------|:------:|
|A| A's skill level|
|B | B's skill level|
|C | C's skill level|
|AvB | the outcome of A vs. B <br> (0 = A wins, 1 = B wins, 2 = tie)|
|BvC | the outcome of B vs. C <br> (0 = B wins, 1 = C wins, 2 = tie)|
|CvA | the outcome of C vs. A <br> (0 = C wins, 1 = A wins, 2 = tie)|


Use the following name attributes:

>- "A"
>- "B"
>- "C"  
>- "AvB"
>- "BvC"
>- "CvA"


Assume that each team has the following prior distribution of skill levels:

|skill level|P(skill level)|
|----|:----:|
|0|0.15|
|1|0.45|
|2|0.30|
|3|0.10|

In addition, assume that the differences in skill levels correspond to the following probabilities of winning:

| skill difference <br> (T2 - T1) | T1 wins | T2 wins| Tie |
|------------|----------|---|:--------:|
|0|0.10|0.10|0.80|
|1|0.20|0.60|0.20|
|2|0.15|0.75|0.10|
|3|0.05|0.90|0.05|

You can check your network implementation in the command line with

```
python probability_tests.py ProbabilityTests.test_games_network
```

### 2b: Calculate posterior distribution for the 3rd match.

_[5 points]_

Suppose that you know the following outcome of two of the three games: A beats B and A draws with C. Calculate the posterior distribution for the outcome of the **BvC** match in `calculate_posterior()`. 

Use the **VariableElimination** provided to perform inference.

You can check your posteriors in the command line with

```
python probability_tests.py ProbabilityTests.test_posterior
```

**NOTE: In the following sections, we'll be arriving at the same values by using sampling.**

#### Hints Regarding sampling for Part 2c, 2d, and 2e

*Hint 1:* In both Metropolis-Hastings and Gibbs sampling, you'll need access to each node's probability distribution and nodes. 
You can access these by calling: 

    A_cpd = bayes_net.get_cpds('A')      
    team_table = A_cpd.values
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values

*Hint 2:*  While performing sampling, you will have to generate your initial sample by sampling uniformly at random an outcome for each non-evidence variable and by keeping the outcome of your evidence variables (`AvB` and `CvA`) fixed.

*Hint 3:* You'll also want to use the random package, e.g. `random.randint()` or `random.choice()`, for the probabilistic choices that sampling makes.

*Hint 4:* In order to count the sample states later on, you'll want to make sure the sample that you return is hashable. One way to do this is by returning the sample as a tuple.



### 2c: Gibbs sampling
_[15 points]_

Implement the Gibbs sampling algorithm, which is a special case of Metropolis-Hastings. You'll do this in `Gibbs_sampler()`, which takes a Bayesian network and initial state value as a parameter and returns a sample state drawn from the network's distribution. In case of Gibbs, the returned state differs from the input state at at-most one variable (randomly chosen).

The method should just consist of a single iteration of the algorithm. If an initial value is not given (initial state is None or and empty list), default to a state chosen uniformly at random from the possible states.

Note: **DO NOT USE the given inference engines or `pgmpy` samplers to run the sampling method**, since the whole point of sampling is to calculate marginals without running inference. 


     "YOU WILL SCORE 0 POINTS ON THIS ASSIGNMENT IF YOU USE THE GIVEN INFERENCE ENGINES FOR THIS PART"


You may find [this](http://gandalf.psych.umn.edu/users/schrater/schrater_lab/courses/AI2/gibbs.pdf) helpful in understanding the basics of Gibbs sampling over Bayesian networks. 


### 2d: Metropolis-Hastings sampling

_[15 points]_

Now you will implement the independent Metropolis-Hastings sampling algorithm in `MH_sampler()`, which is another method for estimating a probability distribution.
The general idea of MH is to build an approximation of a latent probability distribution by repeatedly generating a "candidate" value for each sample vector comprising of the random variables in the system, and then probabilistically accepting or rejecting the candidate value based on an underlying acceptance function. Unlike Gibbs, in case of MH, the returned state can differ from the initial state at more than one variable.
This [slide deck](https://github.gatech.edu/omscs6601/assignment_3/blob/master/resources/mh%20sampling.pdf)
and this [cheat sheet](http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf) provides a nice intro.

This method should just perform a single iteration of the algorithm. If an initial value is not given, default to a state chosen uniformly at random from the possible states. 

Note: **DO NOT USE the given inference engines to run the sampling method**, since the whole point of sampling is to calculate marginals without running inference. 


     "YOU WILL SCORE 0 POINTS IF YOU USE THE PROVIDED INFERENCE ENGINES, OR ANY OTHER SAMPLING METHOD"

### 2e: Comparing sampling methods

_[19 points]_

Now we are ready for the moment of truth.

Given the same outcomes as in 2b, A beats B and A draws with C, you should now estimate the likelihood of different outcomes for the third match by running Gibbs sampling until it converges to a stationary distribution. 
We'll say that the sampler has converged when, for "N" successive iterations, the difference in expected outcome for the 3rd match differs from the previous estimated outcome by less than "delta". `N` is a positive integer, `delta` goes from `(0,1)`. For the most stationary convergence, `delta` should be very small. `N` could typically take values like 10,20,...,100 or even more.

Use the functions from 2c and 2d to measure how many iterations it takes for Gibbs and MH to converge to a stationary distribution over the posterior. See for yourself how close (or not) this stable distribution is to what the Inference Engine returned in 2b. And if not, try tuning those parameters(N and delta). (You might find the concept of "burn-in" period useful). 

You can choose any N and delta (with the bounds above), as long as the convergence criterion is eventually met. For the purpose of this assignment, we'd recommend using a delta approximately equal to 0.001 and N at least as big as 10. 

Repeat this experiment for Metropolis-Hastings sampling.

Fill in the function `compare_sampling()` to perform your experiments

Which algorithm converges more quickly? By approximately what factor? For instance, if Metropolis-Hastings takes twice as many iterations to converge as Gibbs sampling, you'd say that Gibbs converged faster by a factor of 2. Fill in `sampling_question()` to answer both parts.
 
### 2f: Return your name

_[1 point]_

A simple task to wind down the assignment. Return your name from the function aptly called `return_your_name()`.
