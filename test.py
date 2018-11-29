import torch
import torch.optim as optim

import numpy as np
import os
from time import sleep

from common.utils import load_model
from model import DQN, Policy

def test(env, args): 
    p1_current_model = DQN(env, args).to(args.device)
    p2_current_model = DQN(env, args).to(args.device)
    p1_policy = Policy(env).to(args.device)
    p2_policy = Policy(env).to(args.device)
    p1_current_model.eval(), p2_current_model.eval()
    p1_policy.eval(), p2_policy.eval()

    load_model(models={"p1": p1_current_model, "p2": p2_current_model},
               policies={"p1": p1_policy, "p2": p2_policy}, args=args)

    p1_reward_list = []
    p2_reward_list = []
    length_list = []

    for _ in range(30):
        (p1_state, p2_state) = env.reset()
        p1_episode_reward = 0
        p2_episode_reward = 0
        episode_length = 0
        while True:
            if args.render:
                env.render()
                sleep(0.01)

            # Agents follow average strategy
            p1_action = p1_policy.act(torch.FloatTensor(p1_state).to(args.device))
            p2_action = p2_policy.act(torch.FloatTensor(p2_state).to(args.device))

            actions = {"1": p1_action, "2": p2_action}

            (p1_next_state, p2_next_state), reward, done, _ = env.step(actions)

            (p1_state, p2_state) = (p1_next_state, p2_next_state)
            p1_episode_reward += reward[0]
            p2_episode_reward += reward[1]
            episode_length += 1

            if done:
                p1_reward_list.append(p1_episode_reward)
                p2_reward_list.append(p2_episode_reward)
                length_list.append(episode_length)
                break
    
    print("Test Result - Length {:.2f} p1/Reward {:.2f} p2/Reward {:.2f}".format(
        np.mean(length_list), np.mean(p1_reward_list), np.mean(p2_reward_list)))
    