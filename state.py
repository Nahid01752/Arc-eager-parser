from collections import deque
import numpy as np


class State:
    """ State represents the current state of the parser and execute transitions """
    def __init__(self, stack, buffer):
        self.stack = deque(stack)
        self.buffer = deque(buffer)
        self.heads = np.full(len(self.buffer) + len(self.stack), -1, dtype=np.int32)
        self.left_most = np.full(len(self.buffer) + len(self.stack), -1, dtype=np.int32)
        self.right_most = np.full(len(self.buffer) + len(self.stack), -1, dtype=np.int32)

    # get dependent list
    def get_dependent(self, head):
        return [i for i, x in enumerate(self.heads) if x == head]

    # execute transitions
    def left_arc(self):
        last_stack = self.stack.pop()
        self.heads[last_stack] = self.buffer[0]
        self.left_most[self.buffer[0]] = min(self.get_dependent(self.buffer[0]))
        self.right_most[self.buffer[0]] = max(self.get_dependent(self.buffer[0]))

    def right_arc(self):
        last_stack = self.stack[-1]
        self.heads[self.buffer[0]] = last_stack
        self.left_most[last_stack] = min(self.get_dependent(last_stack))
        self.right_most[last_stack] = max(self.get_dependent(last_stack))
        self.stack.append(self.buffer.popleft())

    def reduce(self):
        self.stack.pop()

    def shift(self):
        last_buffer = self.buffer.popleft()
        self.stack.append(last_buffer)
