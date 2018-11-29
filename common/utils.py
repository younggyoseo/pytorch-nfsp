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

def print_log(frame, prev_frame, prev_time, reward_list, length_list, rl_loss_list, sl_loss_list):
    fps = (frame - prev_frame) / (time.time() - prev_time)
    avg_reward = np.mean(reward_list)
    avg_length = np.mean(length_list)
    avg_rl_loss = np.mean(rl_loss_list) if len(rl_loss_list) != 0 else 0.
    avg_sl_loss = np.mean(sl_loss_list) if len(sl_loss_list) != 0 else 0.

    print("Frame: {:<8} FPS: {:.2f} Avg. Reward: {:.2f} Avg. Length: {:.2f} Avg. RL Loss: {:.2f} Avg. SL Loss: {:.2f}".format(
        frame, fps, avg_reward, avg_length, avg_rl_loss, avg_sl_loss
    ))

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
        fname += "dqn.pth"
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
