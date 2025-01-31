"""
all return values are in nano seconds
a decorator - measure_exec_time to add to any method, example:

@measure_exec_time
def some_custom_method(arg1: Any) -> int:
    r = arg1
    for i in range(500):
        r += i
    return r

return_value, exec_time = some_custom_method(10)

OUTPUTS:
return_value = 124760
exec_time = 3.0994415283203125e-05

or with a 'with' statement in case of a block of code, example:
with MeasureExecTime as timer:
    a: int = 1
    b: int = 2
    c: str = '123'
    sum = a + b + int(c)

print(timer.exec_time)
"""

from functools import wraps
from time import perf_counter_ns
from typing import Any


def measure_exec_time(func):
    @wraps(func)
    def wrap(*args, **kwargs) -> (Any, float):
        time_start = perf_counter_ns()
        result = func(*args, **kwargs)
        # print('func:%r args:[%r, %r] took: %2.4f sec' % \
        #       (func.__name__, args, kwargs, time_end - time_start))
        return result, perf_counter_ns() - time_start
    return wrap


class MeasureExecTime:
    def __enter__(self):
        self.start = perf_counter_ns()
        return self

    def __exit__(self, type, value, traceback):
        self.exec_time = perf_counter_ns() - self.start
        # self.readout = f'Time: {self.time:.3f} seconds'
        # print(self.readout)