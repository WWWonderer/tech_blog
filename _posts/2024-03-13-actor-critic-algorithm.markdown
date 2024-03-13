---
title: "Actor critic algorithm"
date: 2024-03-13
categories: "reinforcement_learning"
---

**Relationship with REINFORCE**

Just like REINFORCE of the previous post, actor critic is another policy gradient based method. It can be thought of as a more refined version of REINFORCE, and has the following 2 advantages over the former: 1. Introducing bootstrapping so we no longer need Monte Carlo sampling of the whole episode 2. reduce variance during training. Both speeds up the learning speed of the algorithm.

**Why variance matters and way to reduce it**

Generally speaking, the variance of a random variable X indicates the spread of its values, which can affect the scale and difficulty of the learning task for models. Models often converge more efficiently on data where the input features are scaled similarly, as extreme variations in scale can complicate the optimization process. This is why techniques such as normalization, which adjust the scale of the features to a common range, are highly beneficial for empirical model training. Say we have two random variables $X$ and $Y$ with means $E(X) = \bar{X} = 0$ and ($E(Y) = \bar{Y}) \ne 0$, and we want to estimate $E(YX)$ using $N$ samples, we can do so in two ways:

1. directly estimating $E(YX)$ with $E(YX) = \frac{1}{N} \sum_{i=1}^N Y_i X_i$
2. estimate an equivalent random variable $E[(Y-\bar{Y})X]$ with $\frac{1}{N}\sum_{i=1}^N(Y_i - \bar{Y})X_i$
(as $E[(Y-\bar{Y})X)] = E[(Y-\bar{Y})X] + 0 = E[(Y-\bar{Y})X] + \bar{Y} E(X) = E[(Y-\bar{Y})X + \bar{Y}X] = E(YX)$)

It is easy to see that the 2nd way involves samples with a lower variance, where the variable $Y$ is somewhat normalized by its mean $\bar{Y}$. 

Back to REINFORCE, where we have the following formula for our reward function: $$\nabla_\theta J(\theta) = \mathbb{E}_\pi[Q^\pi(s, a)\nabla_\theta \ln  \pi_\theta(a \vert s)]$$, where $Q^\pi(s, a)\nabla_\theta\ln  \pi_\theta(a \vert s)$ can be thought of as the product of 2 random variables $YX$ above, with $Y = Q^\pi(s, a)$ and $X = \nabla_\theta \ln \pi_\theta(a\|s)$. We can then apply the above technique to reduce sampling variance by doing $Q - \bar{Q}$. A natural candidate for $\bar{Q}$ is the value function $V^\pi(s)$, and therefore we can update REINFORCE to be:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi[(Q^\pi(s, a) - V^\pi(s))\nabla_\theta \ln \pi_\theta(a \vert s)]$$

This is called REINFORCE with baseline, with baseline equal to $V^\pi(s)$ in this case. What is interesting to point out is that the choice of the baseline $B(s)$ does not affect the expected value as long as it is independent of the action $a$ taken:

$$
\begin{aligned}
\mathbb{E}_\pi[(Q^\pi(s, a) - B(s))\nabla_\theta \ln \pi_\theta(a \vert s)] &= \mathbb{E}_\pi[Q^\pi(s, a)\nabla_\theta \ln \pi_\theta(a \vert s)] - \mathbb{E}_\pi[B(s) \nabla_\theta \ln \pi_\theta(a \vert s)] \\
&= \mathbb{E}_\pi[Q^\pi(s, a)\nabla_\theta \ln \pi_\theta(a \vert s)] - \sum_s  \sum_a B(s) \nabla_\theta \ln \pi_\theta(a \vert s) & \text{;Where $s$ and $a$ follow policy $\pi$}\\
&= \mathbb{E}_\pi[Q^\pi(s, a)\nabla_\theta \ln \pi_\theta(a \vert s)] - \sum_s   B(s) \sum_a \nabla_\theta \ln \pi_\theta(a \vert s)\\
&= \mathbb{E}_\pi[Q^\pi(s, a)\nabla_\theta \ln \pi_\theta(a \vert s)] - \sum_s   B(s) 0 & \text{;by the EGLP Lemma}\\
&= \mathbb{E}_\pi[Q^\pi(s, a)\nabla_\theta \ln \pi_\theta(a \vert s)]
\end{aligned}
$$

**Bootstrapping**

Say we take $V^\pi(s)$ as the baseline, it can reduce variance and facilitate convergence, but we still have to sample through a full episode in a Monte Carlo fashion. Another improvement we can do is to use $V^\pi(s)$ itself as an evaluation measure, and stop prematurely during the episode similar to the temporal difference algorithms. This way, we can hope to facilitate learning even further. In fact, instead of generating a whole episode following a certain policy before adjusting/learning it, we will use the policy to generate our trajectory while adjusting/learning it at the same time. Moreover, we will also be learning a value function that further helps us navigate the trajectory. The key idea here is to leverage the next return(s) from the next step(s) ($G_{t:t+1}$ or $G_{t:t+n}$) (bootstrap) immediately instead of using the full expected return ($G_t$) until the end of the episode, while using our value function $V^\pi(s)$ as judge. The true value function $V^\pi(s)$ is surely unknown, but we can approximate it using a differentiable function $\hat{v}(S_t, w)$ parameterized by $w$. Taking a 1 step bootstrap as example, we have:

$$
\begin{aligned}
\theta_{t+1} &= \theta_t + \alpha(G_{t:t+1} - \hat{v}(S_t, w))\frac{\nabla \pi(A_t \vert S_t, \theta_t)}{\pi(A_t \vert S_t, \theta_t)} \\
&= \theta_t + \alpha(R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w))\frac{\nabla \pi(A_t \vert S_t, \theta_t)}{\pi(A_t \vert S_t, \theta_t)} 
\end{aligned}
$$

Where $\gamma$ is an optional discount factor. Below are the REINFORCE and actor-critic algorithms side by side:

> Algorithm: REINFORCE  
> Input: a differentiable policy $\pi$  
parameter: step size $\alpha > 0$
>
loop forever (for each episode):  
$\quad$ Generate an episode $S_0$, $A_0$, $R_1$, ... $S_{T-1}$, $A_{T-1}$, $R_T$, following policy $\pi$  
$\qquad$ Loop for each step of the episode $t = 0, 1, ..., T - 1$:  
$\quad$ $\qquad$ $G \leftarrow \sum_{k=t+1}^T \gamma^{k-t-1} R_k$  
$\quad$ $\qquad$ $\theta \leftarrow \theta + \alpha \gamma^t G \nabla \ln\pi(A_t\vert S_t, \theta)$  


> Algorithm: Actor-Critic  
> Input: a differentiable policy $\pi_\theta$ and a differentiable value function $\hat{v}_w$  
> parameters: step sizes $\alpha^\theta$ and $\alpha^w$   
>
loop forever (for each episode):  
$\quad$ Initialize S (first state of episode)  
$\quad$ $I \leftarrow 1$  
$\quad$ loop while S is not terminal (for each time step):  
$\qquad$ $A \sim \pi(.\vert S, \theta)$  
$\qquad$ Take action $A$, observe next state $S'$ and reward $R$  
$\qquad$ $\delta \leftarrow R + \gamma \hat{v}(S',w) - \hat{v}(S, w)$        (if $S'$ is terminal, then $\hat{v}(S', w) = 0$)  
$\qquad$ $w \leftarrow w + \alpha^w \delta \nabla \hat{v}(S, w)$  
$\qquad$ $$\theta \leftarrow \theta + \alpha^\theta I \delta \nabla \ln \pi(A\vert S, \theta)$$  
$\qquad$ $I \leftarrow \gamma I$  
$\qquad$ $S \leftarrow S'$  

the $I$ and $\gamma$ serve as optional discount factor putting more weight into immediate rewards, but the main difference between the algorithms resides in the following 2 points: 

1. in REINFORCE, we first generate an episode using a policy $\pi$ before doing updates; in Actor-Critic, we generate the steps of the episode using policy $\pi$, and perform updates during each step.  

2. in REINFORCE, only the policy $\pi$ is parameterized and learned; in Actor-Critic, we have both the policy $\pi$ and a value function $\hat{v}$ learning at the same time. The reason why $\hat{v}$ converges this way is similar to why it converges in other temporal difference algorithms such as SARSA. 

With both variance reduction and bootstrapping, we do expect the Actor-Critic algorithm to be much more performant than REINFORCE, which we will verify with an implementation.

**Implementation**

The Pytorch library has also conveniently provided an example for actor-critic for the same cart pole problem as the last post, which can be found [here][actor-critic_implementation]. However, upon closer inspection, the code they provided does not match the pseudocode on Sutton's book. The key difference being they still sampled a whole episode in a Monte Carlo fashion before performing updates. This is not what we want, and thus I implemented the Sutton book version myself: 

{% highlight python %}
import argparse
import gymnasium as gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v1', render_mode='human') if args.render else gym.make('CartPole-v1')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.ffn1 = nn.Linear(4, 128)
        self.ffn2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = F.relu(self.ffn1(x))
        action_prob = F.softmax(self.ffn2(x), dim=-1)
        return action_prob
    
class Valuefn(nn.Module):
    def __init__(self):
        super(Valuefn, self).__init__()
        self.ffn1 = nn.Linear(4, 128)
        self.ffn2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.ffn1(x))
        state_values = self.ffn2(x)
        return state_values

policy = Policy()
valuefn = Valuefn()
optimizer = optim.Adam(list(policy.parameters()) + list(valuefn.parameters()), lr=0.001)

def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def update_policy_and_valuefn(prev_state, state, R, logprob, I, done):
    state_value = torch.tensor(0.0) if done else valuefn(torch.tensor(state))
    prev_state_value = valuefn(torch.tensor(prev_state))
    delta = R + args.gamma * state_value - prev_state_value
    
    # calculate losses
    value_loss = delta.pow(2)
    policy_loss = I * -logprob * delta.detach()
    # update parameters
    loss = value_loss + policy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    running_reward = 10

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0
        I = 1

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):
            # select action from policy
            action, logprob = select_action(state)
            # register previous state
            prev_state = state
            # take the action
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
            # perform updates
            update_policy_and_valuefn(prev_state, state, reward, logprob, I, done)
            I = I * args.gamma
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

if __name__ == '__main__':
    main()
{% endhighlight %}

In this implementation, updates to the policy and the value function are performed throughout the episode instead of after the episode ends. As Pytorch requires a loss function to perform automatic backpropagation, the ideas $$w \leftarrow w + \alpha^w \delta \nabla \hat{v}(S, w)$$ and $$\theta \leftarrow \theta + \alpha^\theta I \delta \nabla \ln \pi(A\vert S, \theta)$$ of the algorithm are roughly translated into 2 loss functions: `value_loss = delta.pow(2)` and `policy_loss = I * -logprob * delta.detach()` respectively. The `detach()` is intended to make the backpropagation for the policy separate from that for the value function. With the default seed, the program solved the cart pole in 550 episodes, and there are quite a lot of variations during training. The REINFORCE algorithm solved the problem in 580 episodes, the difference really isn't significant for this problem. On the other hand, Pytorch's example implementation solved the problem in only 70 episodes, although they sampled in an episodic fashion (not using bootstrapping) and did not separate the optimization of the policy from that of the value function. Maybe for problems with short episodes such as this one, the effect of bootstrapping isn't pronounced and can even be detrimental, and we need to make judgments based on the target problem's characteristics with regard to our implementations.

Reference: [Reinforcement Learning - An Introduction][sutton_book] by Richard S. Sutton and Andrew G. Barto (chapter 13)


[actor-critic_implementation]: https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
[sutton_book]: https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf 

