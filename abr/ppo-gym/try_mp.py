import torch.multiprocessing as mp
from multiprocessing import Queue, Event
# from queue import Queue
import time
import torch


def reader(queue, queue2, model):
    # print(queue.qsize())
    while True:
        msg = queue.get()
        print(msg)
        if msg == 'Done':
            print('reader is reading')
            model = 20
            print('new model value', model)
            break


def writer(queue, queue2, count, model):
    for i in range(count):
        queue.put(i)
    model = 10

        # print(i)
    queue.put('Done')


if __name__ == "__main__":
    torch.set_num_threads(1)
    model = mp.Value('i', 0)
    queue = Queue()
    queue2 = Queue()
    read_p = mp.Process(target=reader, args=(queue, queue2, model))
    start = time.time()
    read_p.start()
    writer(queue, queue2, 100, model)
    read_p.join()
    end = time.time()
    print(end - start)

    update_event = Event()
    update_event.set()
    print(update_event.is_set())
    update_event.clear()
    print(update_event.is_set())

