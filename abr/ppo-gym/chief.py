from model import ActorModel, CriticModel
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


def chief(args, actor, critic, update_event, rolling_event, queue, counter, queue_size,
          actor_optimizer, critic_optimizer):
    s_dim = args.s_dim
    a_dim = args.a_dim
    while True:
        update_event.wait()  # wait while get batch of data
        actor_old = ActorModel(s_dim, a_dim)
        critic_old = CriticModel(s_dim)
        actor_old.load_state_dict(actor.state_dict())  # update old actor parameters
        critic_old.load_state_dict(critic.state_dict())  # update old critic parameters
        data = [queue.get() for _ in range(queue_size.get())]  # receive collected data from workers
        data = np.vstack(data)
        states, actions, returns = data[:, :s_dim], data[:, s_dim:s_dim+1], data[:, -1:]
        states = Variable(torch.FloatTensor(states))
        returns = Variable(torch.FloatTensor(returns))
        values = critic(states)
        advantages = returns - values
        actions = Variable(torch.LongTensor(actions))
        # advantages = Variable(torch.FloatTensor(advantages))
        # print('chief get data')
        # print(states)
        # print(actions)
        # print(advantages)
        for _ in range(args.update_steps):
            # update actor and critic
            logit = actor(states)
            log_probs = F.log_softmax(logit)
            # print('log_probs', log_probs)
            action_log_probs = log_probs.gather(1, actions)
            # print(action_log_probs)
            old_logit = actor_old(states)
            old_log_probs = F.log_softmax(old_logit)
            old_action_log_probs = old_log_probs.gather(1, actions)
            ratio = action_log_probs / old_action_log_probs + 1e-5
            # print('ratio', ratio)
            # ratio = torch.exp(action_log_probs / old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = advantages.pow(2).mean()
            print('loss', actor_loss, critic_loss)
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)
            actor_optimizer.step()
            critic_optimizer.step()
            # print('update')
        # print('update finished')
        # updating finished
        update_event.clear()
        counter.reset()
        queue_size.reset()
        rolling_event.set()
