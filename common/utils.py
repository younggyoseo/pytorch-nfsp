import math
import os
import datetime
import time
import pathlib
import random

import torch
import numpy as np

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def epsilon_scheduler(eps_start, eps_final, eps_decay):
    def function(frame_idx):
        return eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)
    return function

def create_log_dir(args):
    log_dir = ""
    log_dir = log_dir + "{}-".format(args.env)
    if args.negative:
        log_dir = log_dir + "negative-"
    if args.multi_step != 1:
        log_dir = log_dir + "{}-step-".format(args.multi_step)
    if args.dueling:
        log_dir = log_dir + "dueling-"
    log_dir = log_dir + "dqn-{}".format(args.save_model)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = log_dir + now
    log_dir = os.path.join("runs", log_dir)
    return log_dir

def print_log(frame, prev_frame, prev_time, rewards, length_list, rl_losses, sl_losses):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    p1_avg_reward, p2_avg_reward = (np.mean(rewards[0]), np.mean(rewards[1]))
    if len(rl_losses[0]) != 0:
        p1_avg_rl_loss, p2_avg_rl_loss = (np.mean(rl_losses[0]), np.mean(rl_losses[1]))
        p1_avg_sl_loss, p2_avg_sl_loss = (np.mean(sl_losses[0]), np.mean(sl_losses[1]))
    else:
        p1_avg_rl_loss, p2_avg_rl_loss = 0., 0.
        p1_avg_sl_loss, p2_avg_sl_loss = 0., 0.

    avg_length = np.mean(length_list)

    print("Frame: {:<8} FPS: {:.2f} Avg. Tagging Interval Length: {:.2f}".format(frame, fps, avg_length))
    print("Player 1 Avg. Reward: {:.2f} Avg. RL/SL Loss: {:.2f}/{:.2f}".format(
        p1_avg_reward, p1_avg_rl_loss, p1_avg_sl_loss))
    print("Player 2 Avg. Reward: {:.2f} Avg. RL/SL Loss: {:.2f}/{:.2f}".format(
        p2_avg_reward, p2_avg_rl_loss, p2_avg_sl_loss))

def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

def save_model(models, policies, args):
    fname = ""
    fname += "{}-".format(args.env)
    if args.negative:
        fname += "negative-"
    if args.multi_step != 1:
        fname += "{}-step-".format(args.multi_step)
    if args.dueling:
        fname += "dueling-"
    fname += "dqn-{}.pth".format(args.save_model)
    fname = os.path.join("models", fname)

    pathlib.Path('models').mkdir(exist_ok=True)
    torch.save({
        'p1_model': models['p1'].state_dict(),
        'p2_model': models['p2'].state_dict(),
        'p1_policy': policies['p1'].state_dict(),
        'p2_policy': policies['p2'].state_dict(),
    }, fname)

def load_model(models, policies, args):
    if args.load_model is not None:
        fname = os.path.join("models", args.load_model)
        fname += ".pth"
    else:
        fname = ""
        fname += "{}-".format(args.env)
        if args.negative:
            fname += "negative-"
        if args.multi_step != 1:
            fname += "{}-step-".format(args.multi_step)
        if args.dueling:
            fname += "dueling-"
        fname += "dqn-{}.pth".format(args.save_model)
        fname = os.path.join("models", fname)

    # Hack to load models saved with GPU
    if args.device == torch.device("cpu"):
        map_location = lambda storage, loc: storage
    else:
        map_location = None
    
    if not os.path.exists(fname):
        raise ValueError("No model saved with name {}".format(fname))

    checkpoint = torch.load(fname, map_location)
    models['p1'].load_state_dict(checkpoint['p1_model'])
    models['p2'].load_state_dict(checkpoint['p2_model'])
    policies['p1'].load_state_dict(checkpoint['p1_policy'])
    policies['p2'].load_state_dict(checkpoint['p2_policy'])


def set_global_seeds(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)
