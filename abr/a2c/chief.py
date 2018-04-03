from model import Model
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def chief(args, model, update_events, rolling_events, state_queue, queue, counter, queue_size):
    def sample_generator(states, actions, returns, advantages, old_action_logs, batch_size, num_mini_batch):
        # states = torch.FloatTensor(states)
        # actions = torch.FloatTensor(actions)
        # advantages = torch.FloatTensor(advantages)
        # old_action_logs = torch.FloatTensor(old_action_logs)

        mini_batch_size = int(batch_size / num_mini_batch)
        # sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
        indices = np.arange(0, num_mini_batch, 1) * mini_batch_size
        print('indices', indices)
        for indice in indices:
            # indices = torch.LongTensor(indices)
            state_batch = states[indice: indice + mini_batch_size]
            action_batch = actions[indice: indice + mini_batch_size]
            return_batch = returns[indice: indice + mini_batch_size]
            adv_batch = advantages[indice: indice + mini_batch_size]
            old_action_log_batch = old_action_logs[indice: indice + mini_batch_size]

            yield state_batch, action_batch, return_batch, adv_batch, old_action_log_batch, mini_batch_size

    epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=args.a_lr)

    while True:
        epoch += 1
        for rank in range(args.num_processes):
            update_events[rank].wait()  # wait while get batch of data
        model_old = Model()
        model_old.load_state_dict(model.state_dict())  # update old actor parameters
        print('chief queue_size:', queue_size.get())
        data = [queue.get() for _ in range(queue_size.get())]  # receive collected data from workers
        # data = [queue.get() for _ in range(queue.qsize())]
        data = np.vstack(data)
        state_data = [state_queue.get() for _ in range(queue_size.get())]
        # state_data = [state_queue.get() for _ in range(state_queue.qsize())]
        queue_size.reset()
        states = []
        for worker_states in state_data:
            for state in worker_states:
                states.append(state)
        actions, returns, advantages, old_action_logs = data[:, 0:1], data[:, 1:2], data[:, 2:3], data[:, 3:4]
        # states = Variable(torch.FloatTensor(states))
        # returns = Variable(torch.FloatTensor(returns))
        # print(states.shape, actions.shape, returns.shape, advantages.shape)
        # batch_size = returns.shape[0]

        # actions = Variable(torch.LongTensor(actions))
        print('chief get data')
        # print(states)
        # print(actions)
        # print(advantages)
        batch_size = len(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        for sample in sample_generator(states, actions, returns, advantages, old_action_logs, batch_size, args.num_mini_batch):
            state_batch, action_batch, return_batch, adv_batch, old_action_log_batch, mini_batch_size = sample

            logit, values = model(Variable(torch.FloatTensor(state_batch)), batch_size=mini_batch_size)
            probs = F.softmax(logit)
            log_probs = F.log_softmax(logit)
            dist_entropy = -(log_probs * probs).sum(-1).mean()
            # action_log_probs = log_probs.gather(1, Variable(torch.LongTensor(action_batch)))
            # adv_batch = Variable(torch.FloatTensor(adv_batch))
            # ratio = torch.exp(action_log_probs - Variable(torch.FloatTensor(old_action_log_batch)))
            # surr1 = ratio * adv_batch
            # surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_batch
            # action_loss = torch.min(surr1, surr2).mean()
            #
            # value_loss = (Variable(torch.FloatTensor(return_batch)) - values).pow(2).mean()
            #
            adv_batch = Variable(torch.FloatTensor(adv_batch))
            value_loss = adv_batch.pow(2).mean()
            action_loss = -(Variable(torch.FloatTensor(old_action_log_batch)) * adv_batch).mean()

            optimizer.zero_grad()
            (action_loss + value_loss + dist_entropy * 0.01).backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()
            print('update')


        # for _ in range(args.update_steps):
        #     # update actor and critic
        #     logit, values = model(states, batch_size=batch_size)
        #     advantages = returns - values
        #     adv_mean = advantages.mean()
        #     adv_std = advantages.std()
        #     advantages = (advantages - adv_mean) / (adv_std + 1e-5)
        #     print('adv', values.mean())
        #     probs = F.softmax(logit)
        #     log_probs = F.log_softmax(logit)
        #     dist_entropy = -(log_probs * probs).sum(-1).mean()
        #     # print('log_probs', log_probs)
        #     action_log_probs = log_probs.gather(1, actions)
        #     # print(action_log_probs)
        #     old_logit, _ = model_old(states, batch_size=batch_size)
        #     old_log_probs = F.log_softmax(old_logit)
        #     old_action_log_probs = old_log_probs.gather(1, actions)
        #     ratio = action_log_probs / (old_action_log_probs + 1e-5)
        #     # print('ratio', ratio)
        #     # ratio = torch.exp(action_log_probs / old_action_log_probs)
        #     surr1 = ratio * advantages
        #     surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * advantages
        #     actor_loss = -torch.min(surr1, surr2).mean()
        #     critic_loss = advantages.pow(2).mean()
        #     print('loss', actor_loss, critic_loss)
        #     total_loss = actor_loss + critic_loss - dist_entropy * 0.01
        #     optimizer.zero_grad()
        #     total_loss.backward(retain_graph=True)
        #     torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
        #     optimizer.step()
        #     print('update')
        print('update finished')
        # updating finished
        for rank in range(args.num_processes):
            update_events[rank].clear()
            rolling_events[rank].set()
        counter.reset()
        queue_size.reset()
        if epoch % 1000 == 0:
            path = 'results-1/actor.pt-' + str(epoch/1000)
            torch.save(model.state_dict(), path)
            print('saved one model in epoch:', epoch)

