import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import random
import numpy as np
from collections import deque

from common.utils import epsilon_scheduler, update_target, print_log, load_model, save_model
from model import DQN, Policy
from storage import ReplayBuffer, ReservoirBuffer

def train(env, args, writer):
    # RL Model for Player 1
    p1_current_model = DQN(env, args).to(args.device)
    p1_target_model = DQN(env, args).to(args.device)
    update_target(p1_current_model, p1_target_model)

    # RL Model for Player 2
    p2_current_model = DQN(env, args).to(args.device)
    p2_target_model = DQN(env, args).to(args.device)
    update_target(p2_current_model, p2_target_model)

    # SL Model for Player 1, 2
    p1_policy = Policy(env).to(args.device)
    p2_policy = Policy(env).to(args.device)

    if args.load_model and os.path.isfile(args.load_model):
        load_model(models={"p1": p1_current_model, "p2": p2_current_model},
                   policies={"p1": p1_policy, "p2": p2_policy}, args=args)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)

    # Replay Buffer for Reinforcement Learning - Best Response
    p1_replay_buffer = ReplayBuffer(args.buffer_size)
    p2_replay_buffer = ReplayBuffer(args.buffer_size)

    # Reservoir Buffer for Supervised Learning - Average Strategy
    # TODO(Aiden): How to set buffer size of SL?
    p1_reservoir_buffer = ReservoirBuffer(args.buffer_size)
    p2_reservoir_buffer = ReservoirBuffer(args.buffer_size)

    # Deque data structure for multi-step learning
    p1_state_deque = deque(maxlen=args.multi_step)
    p1_reward_deque = deque(maxlen=args.multi_step)
    p1_action_deque = deque(maxlen=args.multi_step)

    p2_state_deque = deque(maxlen=args.multi_step)
    p2_reward_deque = deque(maxlen=args.multi_step)
    p2_action_deque = deque(maxlen=args.multi_step)

    # RL Optimizer for Player 1, 2
    p1_rl_optimizer = optim.Adam(p1_current_model.parameters(), lr=args.lr)
    p2_rl_optimizer = optim.Adam(p2_current_model.parameters(), lr=args.lr)

    # SL Optimizer for Player 1, 2
    # TODO(Aiden): Is it necessary to seperate learning rate for RL/SL?
    p1_sl_optimizer = optim.Adam(p1_policy.parameters(), lr=args.lr)
    p2_sl_optimizer = optim.Adam(p2_policy.parameters(), lr=args.lr)

    # Logging
    length_list = []
    p1_reward_list, p1_rl_loss_list, p1_sl_loss_list = [], [], []
    p2_reward_list, p2_rl_loss_list, p2_sl_loss_list = [], [], []
    p1_episode_reward, p2_episode_reward = 0, 0
    episode_length = 0
    prev_time = time.time()
    prev_frame = 1

    # Main Loop
    (p1_state, p2_state) = env.reset()
    for frame_idx in range(1, args.max_frames + 1):
        is_best_response = False
        # TODO(Aiden):
        # Action should be decided by a combination of Best Response and Average Strategy
        if random.random() > args.eta:
            p1_action = p1_policy.act(torch.FloatTensor(p1_state).to(args.device))
            p2_action = p2_policy.act(torch.FloatTensor(p1_state).to(args.device))
        else:
            is_best_response = True
            epsilon = epsilon_by_frame(frame_idx)
            p1_action = p1_current_model.act(torch.FloatTensor(p1_state).to(args.device), epsilon)
            p2_action = p2_current_model.act(torch.FloatTensor(p2_state).to(args.device), epsilon)

        actions = {"1": p1_action, "2": p2_action}
        (p1_next_state, p2_next_state), reward, done, _ = env.step(actions)

        # Save current state, reward, action to deque for multi-step learning
        p1_state_deque.append(p1_state)
        p2_state_deque.append(p2_state)
        
        p1_reward = reward[0] - 1 if args.negative else reward[0]
        p2_reward = reward[1] - 1 if args.negative else reward[1]
        p1_reward_deque.append(p1_reward)
        p2_reward_deque.append(p2_reward)

        p1_action_deque.append(p1_action)
        p2_action_deque.append(p2_action)

        # Store (state, action, reward, next_state) to Replay Buffer for Reinforcement Learning
        if len(p1_state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(p1_reward_deque, args.gamma)
            n_state = p1_state_deque[0]
            n_action = p1_action_deque[0]
            p1_replay_buffer.push(n_state, n_action, n_reward, p1_next_state, np.float32(done))

            n_reward = multi_step_reward(p2_reward_deque, args.gamma)
            n_state = p2_state_deque[0]
            n_action = p2_action_deque[0]
            p2_replay_buffer.push(n_state, n_action, n_reward, p2_next_state, np.float32(done))
        
        # Store (state, action) to Reservoir Buffer for Supervised Learning
        if is_best_response:
            p1_reservoir_buffer.push(p1_state, p1_action)
            p2_reservoir_buffer.push(p2_state, p2_action)

        (p1_state, p2_state) = (p1_next_state, p2_next_state)

        # Logging
        p1_episode_reward += p1_reward
        p2_episode_reward += p2_reward
        episode_length += 1

        # Episode done. Reset environment and clear logging records
        if done or episode_length > args.max_episode_length:
            (p1_state, p2_state) = env.reset()
            p1_reward_list.append(p1_episode_reward)
            p2_reward_list.append(p2_episode_reward)
            length_list.append(episode_length)
            writer.add_scalar("p1/episode_reward", p1_episode_reward, frame_idx)
            writer.add_scalar("p2/episode_reward", p2_episode_reward, frame_idx)
            writer.add_scalar("data/episode_length", episode_length, frame_idx)
            p1_episode_reward, p2_episode_reward, episode_length = 0, 0, 0
            p1_state_deque.clear(), p2_state_deque.clear()
            p1_reward_deque.clear(), p2_reward_deque.clear()
            p1_action_deque.clear(), p2_action_deque.clear()

        if (len(p1_replay_buffer) > args.rl_start and
            len(p1_reservoir_buffer) > args.sl_start and
            frame_idx % args.train_freq == 0):

            # Update Best Response with Reinforcement Learning
            loss = compute_rl_loss(p1_current_model, p1_target_model, p1_replay_buffer, p1_rl_optimizer, args)
            p1_rl_loss_list.append(loss.item())
            writer.add_scalar("p1/rl_loss", loss.item(), frame_idx)

            loss = compute_rl_loss(p2_current_model, p2_target_model, p2_replay_buffer, p2_rl_optimizer, args)
            p2_rl_loss_list.append(loss.item())
            writer.add_scalar("p2/rl_loss", loss.item(), frame_idx)

            # Update Average Strategy with Supervised Learning
            loss = compute_sl_loss(p1_policy, p1_reservoir_buffer, p1_sl_optimizer, args)
            p1_sl_loss_list.append(loss.item())
            writer.add_scalar("p1/sl_loss", loss.item(), frame_idx)

            loss = compute_sl_loss(p2_policy, p2_reservoir_buffer, p2_sl_optimizer, args)
            p2_sl_loss_list.append(loss.item())
            writer.add_scalar("p2/sl_loss", loss.item(), frame_idx)
        

        if frame_idx % args.update_target == 0:
            update_target(p1_current_model, p1_target_model)
            update_target(p2_current_model, p2_target_model)


        # Logging and Saving models
        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, prev_frame, prev_time, p1_reward_list, length_list, p1_rl_loss_list, p1_sl_loss_list)
            print_log(frame_idx, prev_frame, prev_time, p2_reward_list, length_list, p2_rl_loss_list, p2_sl_loss_list)
            p1_reward_list.clear(), p2_reward_list.clear(), length_list.clear()
            p1_rl_loss_list.clear(), p2_rl_loss_list.clear()
            p1_sl_loss_list.clear(), p2_sl_loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()
            save_model(models={"p1": p1_current_model, "p2": p2_current_model},
                       policies={"p1": p1_policy, "p2": p2_policy}, args=args)
        
        # Render if rendering argument is on
        if args.render:
            env.render()

        save_model(models={"p1": p1_current_model, "p2": p2_current_model},
                    policies={"p1": p1_policy, "p2": p2_policy}, args=args)

def compute_sl_loss(policy, reservoir_buffer, optimizer, args):
    state, action = reservoir_buffer.sample(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)

    probs = policy(state)
    probs_with_actions = probs.gather(1, action.unsqueeze(1))
    log_probs = probs_with_actions.log()

    loss = -1 * log_probs.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def compute_rl_loss(current_model, target_model, replay_buffer, optimizer, args):
    state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
    weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    # Q-Learning with target network
    q_values = current_model(state)
    target_next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = target_next_q_values.max(1)[0]
    expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

    # Huber Loss
    loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret