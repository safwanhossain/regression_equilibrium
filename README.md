# Pure Nash Equilibrium for Lp regression

This repo contains experiments for finding pure Nash Equilibriums in linear regression settings. The idea is that each data point
originates from a self-interested agent who can manipulate what they report, in order to ensure the regression hyperplane passes as 
close to them as possible. Or in other words, each agent seeks to submit values to minimize their loss, with no regard for the global
loss. When multiple agents participate in such behaviour, it induces a game between the agents. We consider the pure Nash Equilibrium
of such a game.

While canonical OLS linear regression minimizes 2-norm, we consider linear regression that minimizes an arbitrary p-norm, p > 1. 
p != 2, does not have a closed form solution for the regression hyperplane; however, it is a strictly convex problem and as such, a
convex optimization library is used to solve this. 

Nash Equilibrium is calculated using a best response algorithm. Each agent is given an opportunity to update their reported value and 
play their best response in the current setting. The algorithm terminates when no agent wants to unilaterally update their reported value
anymore, which is by definition a Nash Equilibrium. Computing the best response of an agent is trivial for p=2 - it follows from the 
projection matrix of OLS. For p != 2, we use a binary search technique to empirically find an agent's best response.

## Experiments
In our experiments, we want to see the number of best response updates necessary to reach a Nash Equilibrium. We also observe the social
cost (in GT literature called price of anarchy) of the equilibrium solution. That is the ratio of the MSE of the equilbrium to the MSE of
reporting honestly. We want to see these quatities as a function of N (number of agent), P, and XDIM (the dimension of the x values). We
sweep each parameter and use default values for the rest. Each datapoint is the average of a 1000 simulations. The code is parallelized 
so that the simulations can run on multiple concurrent processes.

To run experiments (synthetic):
  Go to main_parallel and chose which experiments to run 
  python3 main_parallel 

To plot curves
  For PPoA using q=2 and with BR curve and L1 (figures 1a, 1b, 1c, 2a in experiments)
    ./plot_from_csv <csv_file> <plot_name (image will be saved under this name)> <log scale x (T/F)> 
  For PPoA using different q values and no BR curve or L1 (figures 4 in appendix)
    ./plot_from_csv <csv_file> <plot_name (image will be saved under this name)> <log scale x (T/F)> false true
