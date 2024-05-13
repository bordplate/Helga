from multiprocessing import Process, Queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import deque

queue = Queue()


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def plotter(q):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title('Live State Value and Reward')
    state_value_line, = ax.plot([], [], 'r-', label='State Value')
    accumulated_reward_line, = ax.plot([], [], 'b-', label='Reward')
    ax.legend()
    plt.show()

    state_value_data = deque(maxlen=500)
    accumulated_reward_data = deque(maxlen=500)

    while True:
        message = q.get()  # Receive message from the main process
        if message == 'end':
            break
        new_state_value, new_accumulated_reward = message
        state_value_data.append(new_state_value)
        accumulated_reward_data.append(new_accumulated_reward)

        # Update the plot
        x_range = np.arange(len(state_value_data))
        state_value_line.set_data(x_range, state_value_data)
        accumulated_reward_line.set_data(x_range, accumulated_reward_data)

        ax.relim()
        ax.autoscale_view(True, True, True)
        plt.draw()

        if q.empty():
            mypause(0.01)

    plt.close(fig)


def start_plotting():
    p = Process(target=plotter, args=(queue,))
    p.start()


def add_data(state_value, reward):
    queue.put((state_value, reward))


if __name__ == '__main__':
    queue = Queue()
    p = Process(target=plotter, args=(queue,))
    p.start()

    # Simulation or data acquisition
    for i in range(1000):
        queue.put((i, np.random.random()))

    queue.put('end')
    p.join()
