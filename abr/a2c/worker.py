import torch
import torch.nn.functional as F
#import gym
import numpy as np
from torch.autograd import Variable
from env import Environment


def worker(rank, args, model, update_events, rolling_events, state_queue, queue, counter, queue_size,
           all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit):
    torch.manual_seed(args.seed)
    env = Environment(args, all_cooked_time, all_cooked_bw, all_vp_time, all_vp_unit)
    tag = True
    # epoch = 0
    while tag:
        env.reset()
        action = 144
        state, reward, done = env.step(action)
        buffer_s, buffer_a, buffer_r, buffer_v, buffer_log = [], [], [], [], []
        # epoch += 1
        for step in range(args.num_steps):  # perform K steps.
            if not rolling_events[rank].is_set():  # while chief is updating
                rolling_events[rank].wait()  # wait until chief finished updating
                buffer_s, buffer_a, buffer_r, buffer_v, buffer_log = [], [], [], [], []

            # state = Variable(torch.Tensor(state).unsqueeze(0))
            state = Variable(torch.FloatTensor(state))
            logit, value = model(state.view(-1, 11, 8))
            prob = F.softmax(logit, dim=1)
            log_probs = F.log_softmax(logit)

            # action = prob.multinomial().data
            _, action = torch.max(prob, 1)
            # print('prob', prob)
            # print('action', action.data)
            # action = Variable(torch.LongTensor(action))
            # action = action.view(1, -1)
            # print(action.shape, log_probs.shape)
            action_log_prob = log_probs.gather(1, action.view(1, -1))

            action = action.data
            # print('state', state)
            # print('prob', prob)
            # # print('action', action)
            state, reward, done = env.step(action.numpy()[0])
            print('action: ', action.numpy()[0])
            # print('state: ', state)
            # ep_r += reward
            # print(reward)

            buffer_s.append(state)
            buffer_r.append(reward)
            buffer_a.append(action)
            buffer_v.append(value.data)
            buffer_log.append(action_log_prob.data)

            counter.increment()
            counter_val = counter.get()
            if step == args.num_steps - 1 or counter_val >= args.batch_size:
                # while finish K steps or count to the minimum batch size
                # print(state.type())
                state = Variable(torch.FloatTensor(state))
                _, value = model(state.view(-1, 11, 8))
                value = value.data
                # print('value', value)
                returns = []
                for r in reversed(buffer_r):
                    value = r + args.gamma * value  # compute discounted reward
                    returns.append(value)
                    # print(value)
                returns.reverse()
                # print('buffer_v', buffer_v)
                np_v = np.array(buffer_v)
                # print(returns)
                # print('returns', len(returns))
                np_returns = np.array(returns)

                np_advantages = np_returns - np_v
                np_log = np.array(buffer_log)
                # print('np_v', np_v)
                # print('np_returns', np_returns)
                # print('np_advantages', np_advantages)
                ba, br, badv, blog = np.vstack(buffer_a), np.vstack(np_returns), np.vstack(np_advantages), np.vstack(np_log)
                # print(bs)
                # print(ba)
                # print(badv)
                state_queue.put(buffer_s)
                queue.put(np.hstack((ba, br, badv, blog)))  # put data in the queue
                buffer_s, buffer_a, buffer_r, buffer_v, buffer_log = [], [], [], [], []
                # print(np.hstack((bs, ba, badv)).shape)
                queue_size.increment()
                if counter_val >= args.batch_size:
                    rolling_events[rank].clear()  # stop collecting data
                    update_events[rank].set()  # update policy adn value network
                    print('counter is larger than batch_size', counter_val, queue_size.get())
                    break
            # print(step)


# test worker function
if __name__ == '__main__':
    from main import Counter
    from args import Args
    args = Args()
    counter = Counter()
    env = gym.make(args.env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n  # for Discrete object
    # a_dim = env.action_space.shape[0]  # for Box object
    print(s_dim, a_dim)
    # state = env.reset()
    # state = Variable(torch.Tensor(state).unsqueeze(0))
    # print(state.dim())
    # linear = torch.nn.Linear(3, 1)
    # out = linear(state)
    # print(out)

    from model import ActorModel, CriticModel
    actor = ActorModel(s_dim, a_dim)
    critic = CriticModel(s_dim)
    actor.share_memory()
    critic.share_memory()
    # print(actor, critic)

    import torch.multiprocessing as mp
    update_event, rolling_event = mp.Event(), mp.Event()
    update_event.clear()  # not update now
    rolling_event.set()  # start to roll out
    queue = mp.Queue()  # workers put data in this queue
    counter = Counter()
    queue_size = Counter()

    worker(args, actor, critic, update_event, rolling_event, queue, counter, queue_size)
    # t = Variable(torch.randn(5))
    # print(t)
    # prob = F.softmax(t, dim=0)
    # print(prob)
    # action = prob.multinomial().data
    # print(action)





