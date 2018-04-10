import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from env import Environment
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        # if shared_param.grad is not None:
        #     return
        shared_param._grad = param.grad


def train(rank, args, share_model, counter, lock,
          all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit):

    torch.manual_seed(args.seed + rank)
    env = Environment(args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit, random_seed=args.seed + rank)

    model = ActorCritic()
    optimizer = optim.Adam(share_model.parameters(), lr=args.lr)
    model.train()

    state = env.reset()
    state = Variable(torch.FloatTensor(state))
    done = True
    # reward_sum = 0
    episode_length = 0
    while True:
        model.load_state_dict(share_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            logit, value = model(state.view(-1, 11, 8))
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial()
            # _, action = torch.max(prob, 1)
            log_prob = log_prob.gather(1, action.view(1, -1))
            # print('action: ', action)
            # print(action.data.numpy()[0][0])
            state, reward, done, _ = env.step(action.data.numpy()[0][0])
            state = Variable(torch.FloatTensor(state))
            # print('reward', reward)
            done = done or episode_length >= args.max_episode_length
            # reward = max(min(reward, 1), -1)
            # reward_sum += reward
            # print(reward)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()
                state = Variable(torch.FloatTensor(state))

            # state = torch.FloatTensor(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done or episode_length == args.max_update_step:
                # print('rank: ', rank)
                # print('reward: ', reward_sum)
                # reward_sum = 0
                break

        R = torch.zeros(1, 1)
        if not done:
            logit, value = model(state.view(-1, 11, 8))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        optimizer.zero_grad()
        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        ensure_shared_grads(model, share_model)
        optimizer.step()
        print('update', rank)
        # break
