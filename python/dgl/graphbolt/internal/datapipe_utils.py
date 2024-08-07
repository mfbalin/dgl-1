"""DataPipe utilities"""

import threading
import time

from collections import deque
from typing import final, List, Set, Type  # pylint: disable=no-name-in-module

from torch.utils.data import functional_datapipe, IterDataPipe, MapDataPipe
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps

__all__ = [
    "datapipe_graph_to_adjlist",
    "find_dps",
    "replace_dp",
    "traverse_dps",
]

# Copied from:
# https://github.com/pytorch/data/blob/88c8bdc6662f37649b7ea5df0bd90a4b24a56876/torchdata/datapipes/iter/util/prefetcher.py#L19-L20
# Interval between buffer fulfillment checks
PRODUCER_SLEEP_INTERVAL = 0.0001
# Interval between checking items availability in buffer
CONSUMER_SLEEP_INTERVAL = 0.0001


def _get_parents(result_dict, datapipe_graph):
    for k, (v, parents) in datapipe_graph.items():
        if k not in result_dict:
            result_dict[k] = (v, list(parents.keys()))
            _get_parents(result_dict, parents)


def datapipe_graph_to_adjlist(datapipe_graph):
    """Given a DataPipe graph returned by
    :func:`torch.utils.data.graph.traverse_dps` in DAG form, convert it into
    adjacency list form.

    Namely, :func:`torch.utils.data.graph.traverse_dps` returns the following
    data structure:

    .. code::

       {
           id(datapipe): (
               datapipe,
               {
                   id(parent1_of_datapipe): (parent1_of_datapipe, {...}),
                   id(parent2_of_datapipe): (parent2_of_datapipe, {...}),
                   ...
               }
           )
       }

    We convert it into the following for easier access:

    .. code::

       {
           id(datapipe1): (
               datapipe1,
               [id(parent1_of_datapipe1), id(parent2_of_datapipe1), ...]
           ),
           id(datapipe2): (
               datapipe2,
               [id(parent1_of_datapipe2), id(parent2_of_datapipe2), ...]
           ),
           ...
       }
    """

    result_dict = {}
    _get_parents(result_dict, datapipe_graph)
    return result_dict


# Copied from:
# https://github.com/pytorch/data/blob/88c8bdc6662f37649b7ea5df0bd90a4b24a56876/torchdata/dataloader2/graph/utils.py#L16-L35
def find_dps(graph: DataPipeGraph, dp_type: Type[DataPipe]) -> List[DataPipe]:
    r"""
    Given the graph of DataPipe generated by ``traverse_dps`` function, return DataPipe
    instances with the provided DataPipe type.
    """
    dps: List[DataPipe] = []
    cache: Set[int] = set()

    def helper(g) -> None:  # pyre-ignore
        for dp_id, (dp, src_graph) in g.items():
            if dp_id in cache:
                continue
            cache.add(dp_id)
            # Please not use `isinstance`, there is a bug.
            if type(dp) is dp_type:  # pylint: disable=unidiomatic-typecheck
                dps.append(dp)
            helper(src_graph)

    helper(graph)

    return dps


# Copied from:
# https://github.com/pytorch/data/blob/88c8bdc6662f37649b7ea5df0bd90a4b24a56876/torchdata/dataloader2/graph/utils.py#L82-L97
# Given the DataPipe needs to be replaced and the expected DataPipe, return a new graph
def replace_dp(
    graph: DataPipeGraph, old_datapipe: DataPipe, new_datapipe: DataPipe
) -> DataPipeGraph:
    r"""
    Given the graph of DataPipe generated by ``traverse_dps`` function and the
    DataPipe to be replaced and the new DataPipe, return the new graph of
    DataPipe.
    """
    assert len(graph) == 1

    if id(old_datapipe) in graph:
        graph = traverse_dps(new_datapipe)

    final_datapipe = list(graph.values())[0][0]

    for recv_dp, send_graph in graph.values():
        _replace_dp(recv_dp, send_graph, old_datapipe, new_datapipe)

    return traverse_dps(final_datapipe)


# For each `recv_dp`, find if the source_datapipe needs to be replaced by the new one.
# If found, find where the `old_dp` is located in `recv_dp` and switch it to the `new_dp`
def _replace_dp(
    recv_dp, send_graph: DataPipeGraph, old_dp: DataPipe, new_dp: DataPipe
) -> None:
    old_dp_id = id(old_dp)
    for send_id in send_graph:
        if send_id == old_dp_id:
            _assign_attr(recv_dp, old_dp, new_dp, inner_dp=True)
        else:
            send_dp, sub_send_graph = send_graph[send_id]
            _replace_dp(send_dp, sub_send_graph, old_dp, new_dp)


# Recursively re-assign datapipe for the sake of nested data structure
# `inner_dp` is used to prevent recursive call if we have already met a `DataPipe`
def _assign_attr(obj, old_dp, new_dp, inner_dp: bool = False):
    if obj is old_dp:
        return new_dp
    elif isinstance(obj, (IterDataPipe, MapDataPipe)):
        # Prevent recursive call for DataPipe
        if not inner_dp:
            return None
        for k in list(obj.__dict__.keys()):
            new_obj = _assign_attr(obj.__dict__[k], old_dp, new_dp)
            if new_obj is not None:
                obj.__dict__[k] = new_obj
                break
        return None
    elif isinstance(obj, dict):
        for k in list(obj.keys()):
            new_obj = _assign_attr(obj[k], old_dp, new_dp)
            if new_obj is not None:
                obj[k] = new_obj
                break
        return None
    # Tuple is immutable, has to re-create a tuple
    elif isinstance(obj, tuple):
        temp_list = []
        flag = False
        for item in obj:
            new_obj = _assign_attr(item, old_dp, new_dp, inner_dp)
            if new_obj is not None:
                flag = True
                temp_list.append(new_dp)
            else:
                temp_list.append(item)
        if flag:
            return tuple(temp_list)  # Special case
        else:
            return None
    elif isinstance(obj, list):
        for i in range(len(obj)):  # pylint: disable=consider-using-enumerate
            new_obj = _assign_attr(obj[i], old_dp, new_dp, inner_dp)
            if new_obj is not None:
                obj[i] = new_obj
                break
        return None
    elif isinstance(obj, set):
        new_obj = None
        for item in obj:
            if _assign_attr(item, old_dp, new_dp, inner_dp) is not None:
                new_obj = new_dp
                break
        if new_obj is not None:
            obj.remove(old_dp)
            obj.add(new_dp)
        return None
    else:
        return None


class _PrefetchData:
    def __init__(self, source_datapipe, buffer_size: int):
        self.run_prefetcher: bool = True
        self.prefetch_buffer: Deque = deque()
        self.buffer_size: int = buffer_size
        self.source_datapipe = source_datapipe
        self.stop_iteration: bool = False
        self.paused: bool = False


# Copied from:
# https://github.com/pytorch/data/blob/88c8bdc6662f37649b7ea5df0bd90a4b24a56876/torchdata/datapipes/iter/util/prefetcher.py#L34-L172
@functional_datapipe("prefetch")
class PrefetcherIterDataPipe(IterDataPipe):
    r"""
    Prefetches elements from the source DataPipe and puts them into a buffer
    (functional name: ``prefetch``). Prefetching performs the operations (e.g.
    I/O, computations) of the DataPipes up to this one ahead of time and stores
    the result in the buffer, ready to be consumed by the subsequent DataPipe.
    It has no effect aside from getting the sample ready ahead of time.

    This is used by ``MultiProcessingReadingService`` when the arguments
    ``worker_prefetch_cnt`` (for prefetching at each worker process) or
    ``main_prefetch_cnt`` (for prefetching at the main loop) are greater than 0.

    Beyond the built-in use cases, this can be useful to put after I/O DataPipes
    that have expensive I/O operations (e.g. takes a long time to request a file
    from a remote server).

    Args:
        source_datapipe: IterDataPipe from which samples are prefetched
        buffer_size: the size of the buffer which stores the prefetched samples

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(file_paths).open_files().prefetch(5)
    """

    def __init__(self, source_datapipe, buffer_size: int = 10):
        self.source_datapipe = source_datapipe
        if buffer_size <= 0:
            raise ValueError(
                "'buffer_size' is required to be a positive integer."
            )
        self.buffer_size = buffer_size
        self.thread: Optional[threading.Thread] = None
        self.prefetch_data: Optional[_PrefetchData] = None

    @staticmethod
    def thread_worker(
        prefetch_data: _PrefetchData,
    ):  # pylint: disable=missing-function-docstring
        itr = iter(prefetch_data.source_datapipe)
        while not prefetch_data.stop_iteration:
            # Run if not paused
            while prefetch_data.run_prefetcher:
                if (
                    len(prefetch_data.prefetch_buffer)
                    < prefetch_data.buffer_size
                ):
                    try:
                        item = next(itr)
                        prefetch_data.prefetch_buffer.append(item)
                    except Exception as e:  # pylint: disable=broad-except
                        prefetch_data.run_prefetcher = False
                        prefetch_data.stop_iteration = True
                        prefetch_data.prefetch_buffer.append(e)
                else:  # Buffer is full, waiting for main thread to consume items
                    # TODO: Calculate sleep interval based on previous consumption speed
                    time.sleep(PRODUCER_SLEEP_INTERVAL)
            prefetch_data.paused = True
            # Sleep longer when this prefetcher thread is paused
            time.sleep(PRODUCER_SLEEP_INTERVAL * 10)

    def __iter__(self):
        try:
            prefetch_data = _PrefetchData(
                self.source_datapipe, self.buffer_size
            )
            self.prefetch_data = prefetch_data
            thread = threading.Thread(
                target=PrefetcherIterDataPipe.thread_worker,
                args=(prefetch_data,),
                daemon=True,
            )
            thread.start()
            self.thread = thread

            while (
                not prefetch_data.stop_iteration
                or len(prefetch_data.prefetch_buffer) > 0
            ):
                if len(prefetch_data.prefetch_buffer) > 0:
                    data = prefetch_data.prefetch_buffer.popleft()
                    if isinstance(data, Exception):
                        if isinstance(data, StopIteration):
                            break
                        raise data
                    yield data
                else:
                    time.sleep(CONSUMER_SLEEP_INTERVAL)
        finally:
            if "prefetch_data" in locals():
                prefetch_data.run_prefetcher = False
                prefetch_data.stop_iteration = True
                prefetch_data.paused = False
            if "thread" in locals():
                thread.join()

    def __getstate__(self):
        """
        Getting state in threading environment requires next operations:
            1) Stopping of the producer thread.
            2) Saving buffer.
            3) Adding lazy restart of producer thread when __next__ is called again
              (this will guarantee that you only change state of the source_datapipe
               after entire state of the graph is saved).
        """
        # TODO: Update __getstate__ and __setstate__ to support snapshotting and restoration
        return {
            "source_datapipe": self.source_datapipe,
            "buffer_size": self.buffer_size,
        }

    def __setstate__(self, state):
        self.source_datapipe = state["source_datapipe"]
        self.buffer_size = state["buffer_size"]
        self.thread = None

    @final
    def reset(self):  # pylint: disable=missing-function-docstring
        self.shutdown()

    def pause(self):  # pylint: disable=missing-function-docstring
        if self.thread is not None:
            assert self.prefetch_data is not None
            self.prefetch_data.run_prefetcher = False
            if self.thread.is_alive():
                # Blocking until the thread is paused
                while not self.prefetch_data.paused:
                    time.sleep(PRODUCER_SLEEP_INTERVAL * 10)

    @final
    def resume(self):  # pylint: disable=missing-function-docstring
        if (
            self.thread is not None
            and self.prefetch_data is not None
            and (
                not self.prefetch_data.stop_iteration
                or len(self.prefetch_data.prefetch_buffer) > 0
            )
        ):
            self.prefetch_data.run_prefetcher = True
            self.prefetch_data.paused = False

    @final
    def shutdown(self):  # pylint: disable=missing-function-docstring
        if hasattr(self, "prefetch_data") and self.prefetch_data is not None:
            self.prefetch_data.run_prefetcher = False
            self.prefetch_data.stop_iteration = True
            self.prefetch_data.paused = False
            self.prefetch_data = None
        if hasattr(self, "thread") and self.thread is not None:
            self.thread.join()
            self.thread = None

    def __del__(self):
        self.shutdown()

    def __len__(self) -> int:
        if isinstance(self.source_datapipe, Sized):
            return len(self.source_datapipe)
        raise TypeError(
            f"{type(self).__name__} instance doesn't have valid length"
        )
