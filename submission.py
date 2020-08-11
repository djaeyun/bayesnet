import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32, random
#  pgmpy
import pgmpy
import numpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    BayesNet.add_node('alarm')
    BayesNet.add_node('faulty alarm')
    BayesNet.add_node('gauge')
    BayesNet.add_node('faulty gauge')
    BayesNet.add_node('temperature')

    BayesNet.add_edge('temperature', 'faulty gauge')
    BayesNet.add_edge('temperature', 'gauge')
    BayesNet.add_edge('faulty gauge', 'gauge')
    BayesNet.add_edge('gauge', 'alarm')
    BayesNet.add_edge('faulty alarm', 'alarm')
    return BayesNet

def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    cpd_list = [TabularCPD('temperature', 2, values=[[0.80], [0.20]]), 
                TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]]),
                TabularCPD('faulty gauge', 2, values=[[0.95, 0.20], [0.05, 0.80]], evidence=['temperature'], evidence_card = [2]),
                TabularCPD('gauge', 2, values=[[0.95, 0.20, 0.05, 0.80], [0.05, 0.80, 0.95, 0.20]], evidence=['temperature', 'faulty gauge'], evidence_card = [2, 2]),
                TabularCPD('alarm', 2, values=[[0.90, 0.55, 0.10, 0.45], [0.10, 0.45, 0.90, 0.55]], evidence=['gauge', 'faulty alarm'], evidence_card = [2, 2])]
    bayes_net.add_cpds(*cpd_list)
    return bayes_net

def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    prob = marginal_prob['alarm'].values[1]
    return prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""

    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    prob = marginal_prob['gauge'].values[1]
    return prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'], evidence={'alarm': 1, 'faulty alarm':0, 'faulty gauge':0}, joint=False)
    prob = conditional_prob['temperature'].values[1]
    return prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    
    BayesNet.add_node('A')
    BayesNet.add_node('B')
    BayesNet.add_node('C')
    BayesNet.add_node('AvB')
    BayesNet.add_node('BvC')
    BayesNet.add_node('CvA')

    BayesNet.add_edge('A', 'AvB')
    BayesNet.add_edge('B', 'AvB')
    BayesNet.add_edge('B', 'BvC')
    BayesNet.add_edge('C', 'BvC')
    BayesNet.add_edge('A', 'CvA')
    BayesNet.add_edge('C', 'CvA')      

    cpd_list = [TabularCPD('A', 4, values=[[0.15], [0.45], [0.30], [0.10]]),
                TabularCPD('B', 4, values=[[0.15], [0.45], [0.30], [0.10]]),
                TabularCPD('C', 4, values=[[0.15], [0.45], [0.30], [0.10]]),
                TabularCPD('AvB', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], 
                                             [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                             [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], evidence=['A', 'B'], evidence_card = [4, 4]),
                TabularCPD('BvC', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], 
                                             [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                             [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], evidence=['B', 'C'], evidence_card = [4, 4]),
                TabularCPD('CvA', 3, values=[[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10], 
                                             [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                             [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]], evidence=['C', 'A'], evidence_card = [4, 4])]

    BayesNet.add_cpds(*cpd_list)
    return BayesNet

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'], evidence={'AvB':0, 'CvA':2}, joint=False)
    posterior = conditional_prob['BvC'].values
    return posterior

def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = []
    for item in initial_state:
        sample.append(item)

    variables = ['A', 'B', 'C', 'AvB', 'BvC', 'CvA']
    dict = {}
    # Use BayesNet function to return possible values for the six indices. Use AvB: 0 and CvA: 2 as fixed evidence, then randomize the other four
    for i in range(len(sample[0:6])):
        if i == 3:
            sample[i] = 0
        elif i == 5:
            sample[i] = 2
        elif sample[i] is None:
            if i in (0, 1, 2):
                sample[i] = random.choice([0, 1, 2, 3])
            if i == 4:
                sample[i] = random.choice([0, 1, 2])
        dict[variables[i]] = sample[i]

    # Pick an index, ranging from 0 to 5, at random
    index_list = [0, 1, 2, 4]
    index = random.choice(index_list)
    #index = 2
    index_list.remove(index)
    outcomes = [0, 1, 2]
    skill_levels = [0, 1, 2, 3]

    # Suppose chosen is A, calculate P(A | B, C, AvB, BvC, CvA) = P(B, C, AvB, BvC, CvA | A) * P(A) / P(B, C, AvB, BvC, CvA)
    # P(AvB, CvA | A) = P(AvB | A) * P(CvA | A)
    # P(AvB, CvA)     = P(AvB | A) * P(CvA | A) * P(A) + P(AvB | nA) * P(CvA | nA) * P(nA)
    probabilities = []
    v1 = variables[index]

    # Check with posterior method
    '''
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['C'], evidence={'A': dict['A'], 'B': dict['B'], 'AvB':0, 'BvC':dict['BvC'], 'CvA':2}, joint=False)
    print('sample: ', sample)
    print('posterior: ', conditional_prob['C'].values)
    '''
    if v1 == 'A':
        vs_1 = 'AvB'
        vs_2 = 'CvA'
        for skill in skill_levels:
            p_v1 = bayes_net.get_cpds(v1).values[skill]
            p_AvB_v1 = bayes_net.get_cpds(vs_1).values[dict[vs_1]][skill][dict['B']]
            p_CvA_v1 = bayes_net.get_cpds(vs_2).values[dict[vs_2]][dict['C']][skill]
            num = p_AvB_v1 * p_CvA_v1 * p_v1

            den = 0 
            for skill_d in skill_levels:
                p_AvB_vx = bayes_net.get_cpds(vs_1).values[dict[vs_1]][skill_d][dict['B']]
                p_CvA_vx = bayes_net.get_cpds(vs_2).values[dict[vs_2]][dict['C']][skill_d]
                den = den + p_AvB_vx * p_CvA_vx * bayes_net.get_cpds('A').values[skill_d]
            prob = num / den
            probabilities.append(prob)

    elif v1 == 'B':
        vs_1 = 'BvC'
        vs_2 = 'AvB'
        for skill in skill_levels:
            p_v1 = bayes_net.get_cpds(v1).values[skill]
            p_BvC_v1 = bayes_net.get_cpds(vs_1).values[dict[vs_1]][skill][dict['C']]
            p_AvB_v1 = bayes_net.get_cpds(vs_2).values[dict[vs_2]][dict['A']][skill]
            num = p_AvB_v1 * p_BvC_v1 * p_v1

            den = 0
            for skill_d in skill_levels:
                p_BvC_vx = bayes_net.get_cpds(vs_1).values[dict[vs_1]][skill_d][dict['C']]
                p_AvB_vx = bayes_net.get_cpds(vs_2).values[dict[vs_2]][dict['A']][skill_d]
                den = den + p_AvB_vx * p_BvC_vx * bayes_net.get_cpds('B').values[skill_d]
            prob = num / den
            probabilities.append(prob)
        #print(probabilities)
    elif v1 == 'C':
        vs_1 = 'CvA'
        vs_2 = 'BvC'
        for skill in skill_levels:
            p_v1 = bayes_net.get_cpds(v1).values[skill]
            p_CvA_v1 = bayes_net.get_cpds(vs_1).values[dict[vs_1]][skill][dict['A']]
            p_BvC_v1 = bayes_net.get_cpds(vs_2).values[dict[vs_2]][dict['B']][skill]
            num = p_CvA_v1 * p_BvC_v1 * p_v1

            # Calculate p_CvA_nv1 and p_BvC_nv1
            den = 0
            for skill_d in skill_levels:
                p_CvA_vx = bayes_net.get_cpds(vs_1).values[dict[vs_1]][skill_d][dict['A']]
                p_BvC_vx = bayes_net.get_cpds(vs_2).values[dict[vs_2]][dict['B']][skill_d]
                den = den + p_CvA_vx * p_BvC_vx * bayes_net.get_cpds('C').values[skill_d]
            prob = num / den
            probabilities.append(prob)
        #print(probabilities)
    elif v1 == 'BvC':
        # Suppose chosen is BvC, calculate P(BvC | B, C), can look up directly
        for outcome in outcomes:
            prob = bayes_net.get_cpds('BvC').values[outcome][dict['B']][dict['C']]
            probabilities.append(prob)
            
    if index in (0, 1, 2):
        sample[index] = numpy.random.choice(skill_levels, 1, p=probabilities)[0]
    else:
        sample[index] = numpy.random.choice(outcomes, 1, p=probabilities)[0]
    sample = tuple(sample)
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)    
    outcomes = [0, 1, 2]
    skill_levels = [0, 1, 2, 3]
    
    # 1. Select initial value and randomly select if any are None
    sample = []
    for item in initial_state:
        sample.append(item)

    for i in range(len(sample[0:6])):
        if sample[i] is None:
            if i == 3:
                sample[i] = 0
            elif i == 5:
                sample[i] = 2
            elif sample[i] is None:
                if i in (0, 1, 2):
                    sample[i] = random.choice(skill_levels)
                if i == 4:
                    sample[i] = random.choice(outcomes)

    # 2. Draw next candidate based on each distribution
    candidate = []
    outcomes = [0, 1, 2]
    skill_levels = [0, 1, 2, 3]
    for i in range(6):
        if i < 3:
            candidate.append(numpy.random.choice(skill_levels))
        elif i == 3:
            candidate.append(0)
        elif i == 4:
            candidate.append(numpy.random.choice(outcomes))
        else:
            candidate.append(2)

    # 3. Compute alpha
    # P(A, B, C, AvB, BvC, CvA) = P(A) * P(B) * P(C) * P(AvB | A, B) * P(BvC | B, C) * P(CvA | C, A)
    p_A_cand = team_table[candidate[0]]
    p_B_cand = team_table[candidate[1]]
    p_C_cand = team_table[candidate[2]]
    p_AvB_AB_cand = match_table[candidate[3]][candidate[0]][candidate[1]]
    p_BvC_BC_cand = match_table[candidate[4]][candidate[1]][candidate[2]]
    p_CvA_CA_cand = match_table[candidate[5]][candidate[2]][candidate[0]]

    p_A_prior = team_table[sample[0]]
    p_B_prior = team_table[sample[1]]
    p_C_prior = team_table[sample[2]]
    p_AvB_AB_prior = match_table[sample[3]][sample[0]][sample[1]]
    p_BvC_BC_prior = match_table[sample[4]][sample[1]][sample[2]]
    p_CvA_CA_prior = match_table[sample[5]][sample[2]][sample[0]]

    num = p_A_cand * p_B_cand * p_C_cand * p_AvB_AB_cand * p_BvC_BC_cand * p_CvA_CA_cand
    den = p_A_prior * p_B_prior * p_C_prior * p_AvB_AB_prior * p_BvC_BC_prior * p_CvA_CA_prior
    alpha = num / den

    # 4. Accept/Reject
    #print('alpha: ', alpha, candidate)
    #print('prior: ', sample)
    if alpha >= 1:
        sample = candidate
    else:
        choice = numpy.random.choice([0, 1], 1, p=[alpha, 1 - alpha]) 
        if choice == 0:
            sample = candidate
    #print('post: ', sample)
    return sample

def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    delta = 0.000001
    N = 100
    B = 10000

    # Run Gibbs Sampling
    Gibbs_sample = initial_state
    Gibbs_results = [0, 0, 0]
    index = 4
    conv_counter = 0
    diff_list = []
    while conv_counter != N:
        Gibbs_sample = Gibbs_sampler(bayes_net, Gibbs_sample)
        Gibbs_count = Gibbs_count + 1

        if Gibbs_count > B:
            Gibbs_results[Gibbs_sample[index]] = Gibbs_results[Gibbs_sample[index]] + 1
            Gibbs_total = Gibbs_results[0] + Gibbs_results[1] + Gibbs_results[2]
            Gibbs_convergence = [Gibbs_results[0] / Gibbs_total, Gibbs_results[1] / Gibbs_total, Gibbs_results[2] / Gibbs_total]

            diff_list.append((Gibbs_convergence))
            conv_counter = 0
            if Gibbs_count > N + B:
                for i in range(N):
                    conv_avg = (abs(diff_list[-(i+1)][0] - diff_list[-(i+2)][0]) + abs(diff_list[-(i+1)][1] - diff_list[-(i+2)][1]) + abs(diff_list[-(i+1)][2] - diff_list[-(i+2)][2])) / 3
                    if conv_avg < delta:
                        conv_counter = conv_counter + 1

    print(Gibbs_count, Gibbs_convergence)

    # Run MH Sampling
    MH_sample = initial_state
    MH_results = [0, 0, 0]
    index = 4
    conv_counter = 0
    diff_list = []
    while conv_counter != N:
        MH_sample_prior = MH_sample
        MH_sample = MH_sampler(bayes_net, MH_sample)
        MH_count = MH_count + 1
        if MH_sample_prior == MH_sample:
            MH_rejection_count = MH_rejection_count + 1

        if MH_count > B:
            MH_results[MH_sample[index]] = MH_results[MH_sample[index]] + 1
            MH_total = MH_results[0] + MH_results[1] + MH_results[2]
            MH_convergence = [MH_results[0] / MH_total, MH_results[1] / MH_total, MH_results[2] / MH_total]
            #print(MH_count, MH_convergence, MH_rejection_count)

            diff_list.append((MH_convergence))
            conv_counter = 0
            if MH_count > N + B:
                for i in range(N):
                    conv_avg = (abs(diff_list[-(i+1)][0] - diff_list[-(i+2)][0]) + abs(diff_list[-(i+1)][1] - diff_list[-(i+2)][1]) + abs(diff_list[-(i+1)][2] - diff_list[-(i+2)][2])) / 3
                    if conv_avg < delta:
                        conv_counter = conv_counter + 1
    print(MH_count, MH_convergence)
    print(MH_rejection_count)
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def sampling_question():
    """Question about sampling performance."""
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.17
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    name = 'David Jaeyun Kim'
    return name
    raise NotImplementedError