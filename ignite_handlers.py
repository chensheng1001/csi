import logging
import pathlib
import warnings
from typing import Callable, Dict, List, NamedTuple, NoReturn, Optional, Union

import torch
from ignite import engine as ig_engine, handlers as ig_handlers
from ignite.contrib import handlers as ig_c_handlers


class BestPerformance:
    """
    Monitor at which epoch the trained model achieves its best performance.
    
    Records will be kept in form `("score", "epoch")`.
    
    :param score_function: a function taking a single argument, :class:`ignite.engine.Engine` object, and returning a
        score (`float`). Records with highest scores will be retained.
    :param num_kept: How many performance records should be kept.
    :param global_step_transform: To output a desired global step. If provided, uses function output as global_step.
        To setup global step from another engine, please use :meth:`ignite.handlers.global_step_from_engine`.
    """
    
    class Item(NamedTuple):
        """Item."""
        score: float
        epoch: int
        metrics: dict
    
    def __init__(self, score_function: Callable[[ig_engine.Engine], float],
                 global_step_transform: Callable[[ig_engine.Engine, ig_engine.Events], int],
                 num_kept: int = 1):
        self._score_function = score_function
        self._num_kept = num_kept
        self._kept: List[BestPerformance.Item] = []
        self.global_step_transform = global_step_transform
    
    def __call__(self, engine: ig_engine.Engine):
        step = self.global_step_transform(engine, engine.last_event_name)
        score = self._score_function(engine)
        
        if len(self._kept) < self._num_kept or self._kept[-1].score < score:
            if len(self._kept) == self._num_kept:
                self._kept.pop(-1)
            self._kept.append(BestPerformance.Item(score, step, engine.state.metrics.copy()))
            self._kept.sort(key = lambda item: item[0], reverse = True)
    
    @property
    def kept(self) -> List[Item]:
        """
        
        :return: The kept records.
        """
        return self._kept


class BaseMetricLogger:
    r"""
    Base metric logger.
    Partially copied from `BaseOutputHandler`.
    """
    
    def __init__(self, metric_names: Union[str, List[str], None] = None,
                 output_transform: Optional[Callable] = None,
                 global_step_transform: Optional[Callable[[ig_engine.Engine, ig_engine.Events], int]] = None):
        """
        
        :param metric_names: list of metric names to log or a string "all" to log all available metrics.
        :param output_transform : Output transform function to prepare `engine.state.output` as a number.
            This function can also return a dictionary to label the outputs with corresponding keys.
        :param global_step_transform: To output a desired global step. If provided, uses function output as global_step.
            To setup global step from another engine, please use :meth:`ignite.handlers.global_step_from_engine`.
        """
        if metric_names is not None:
            if not (isinstance(metric_names, list) or (isinstance(metric_names, str) and metric_names == "all")):
                raise TypeError("metric_names should be either a list or equal 'all', " "got {} instead.".format(
                        type(metric_names)))
        if output_transform is not None and not callable(output_transform):
            raise TypeError("output_transform should be a function, got {} instead.".format(type(output_transform)))
        if output_transform is None and metric_names is None:
            raise ValueError("Either metric_names or output_transform should be defined")
        if global_step_transform is not None and not callable(global_step_transform):
            raise TypeError(
                    "global_step_transform should be a function, got {} instead.".format(type(global_step_transform)))
        
        self.metric_names = metric_names
        self.output_transform = output_transform
        self.global_step_transform = global_step_transform
    
    def _setup_output_metrics(self, engine: ig_engine.Engine):
        """
        Helper method to retrieve metrics to log.
        """
        metrics = {}
        if self.metric_names is not None:
            if isinstance(self.metric_names, str) and self.metric_names == "all":
                metrics = engine.state.metrics.copy()
            else:
                for name in self.metric_names:
                    if name not in engine.state.metrics:
                        warnings.warn("Provided metric name '{}' is missing "
                                      "in engine's state metrics: {}".format(name, list(engine.state.metrics.keys())))
                        continue
                    metrics[name] = engine.state.metrics[name]
        
        if self.output_transform is not None:
            output_dict = self.output_transform(engine.state.output)
            
            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}
            
            metrics.update({name: value for name, value in output_dict.items()})
        
        return metrics
    
    def _get_step(self, engine: ig_engine.Engine) -> int:
        if self.global_step_transform is not None:
            global_step = self.global_step_transform(engine, engine.last_event_name)
        else:
            global_step = engine.state.get_event_attrib_value(engine.last_event_name)
        return global_step


class MetricsLogger(BaseMetricLogger):
    """
    Print metrics to progress bar and log them to file.
    """
    
    def __init__(self,
                 metric_names: Union[str, List[str], None] = None,
                 output_transform: Optional[Callable] = None,
                 global_step_transform: Optional[Callable[[ig_engine.Engine, ig_engine.Events], int]] = None,
                 progress_bar: Optional[ig_c_handlers.ProgressBar] = None, message_header: Optional[str] = '',
                 file_path: Optional[pathlib.Path] = None):
        """
        
        :param metric_names: list of metric names to log or a string "all" to log all available metrics.
        :param output_transform : Output transform function to prepare `engine.state.output` as a number.
            This function can also return a dictionary to label the outputs with corresponding keys.
        :param global_step_transform: To output a desired global step. If provided, uses function output as global_step.
            To setup global step from another engine, please use :meth:`ignite.handlers.global_step_from_engine`.
        :param progress_bar: The progress bar.
        :param message_header: The header of message printed to progress bar.
        :param file_path: The file path to log metrics to.
        """
        super(MetricsLogger, self).__init__(metric_names, output_transform, global_step_transform)
        
        self._progress_bar = progress_bar
        self._message_header = message_header
        self._file_path = file_path
    
    def __call__(self, engine: ig_engine.Engine):
        if not (self._progress_bar or self._file_path):
            return  # nowhere to log metrics to
        
        step = self._get_step(engine)
        
        metrics = self._setup_output_metrics(engine)
        
        columns = list(metrics.keys())
        
        if self._progress_bar:
            # Prepare metrics values for screen output.
            values = []
            for value in metrics.values():
                if type(value) == torch.Tensor:
                    tmp = value.tolist()
                elif type(value) == float:
                    tmp = round(value, 6)
                else:
                    tmp = value
                values.append(str(tmp))
            
            message = self._message_header.capitalize() + " [{}]".format(step)
            for name, value in zip(columns, values):
                message += " | {name}: {value}".format(name = name, value = value)
            
            self._progress_bar.log_message(message)
        
        if self._file_path:
            # Prepare metrics values for file output.
            values = []
            for value in engine.state.metrics.values():
                if type(value) == torch.Tensor:
                    tmp = value.tolist()
                elif type(value) == float:
                    tmp = value
                else:
                    tmp = value
                values.append(str(tmp))
            
            with self._file_path.open('a') as f:
                if f.tell() == 0:
                    print("\t".join(['epoch'] + columns), file = f)
                print("\t".join([str(step)] + values), file = f)


class HistoryRecorder(BaseMetricLogger):
    """
    Record the history of given metrics.
    """
    
    def __init__(self, metric_names: Union[str, List[str], None] = None,
                 output_transform: Optional[Callable] = None,
                 global_step_transform: Optional[Callable[[ig_engine.Engine, ig_engine.Events], int]] = None):
        """
        
        :param metric_names: list of metric names to record or a string "all" to record all available metrics.
        :param output_transform : Output transform function to prepare `engine.state.output` as a number.
            This function can also return a dictionary to label the outputs with corresponding keys.
        :param global_step_transform: To output a desired global step. If provided, uses function output as global_step.
            To setup global step from another engine, please use :meth:`ignite.handlers.global_step_from_engine`.
        """
        super(HistoryRecorder, self).__init__(metric_names, output_transform, global_step_transform)
        
        self._record: Dict[int, dict] = {}
    
    def __call__(self, engine: ig_engine.Engine):
        step = self._get_step(engine)
        
        metrics = self._setup_output_metrics(engine)
        
        self._record.update({step: metrics})
    
    def reset(self) -> NoReturn:
        """
        Reset recorder.
        """
        self._record = {}
    
    def get(self, start: int, end: int, reduce: Optional[str] = None) -> Union[dict, List[dict]]:
        """
        Return records form global_step start to end (not included).
        
        :param start: Start step.
        :param end: End step (not included).
        :param reduce: 'mean' to return the mean values over steps.
        """
        records = []
        for i in range(start, end):
            try:
                records.append(self._record[i].copy())
            except KeyError:
                raise ValueError("Record of step {} not found.".format(i))
        
        if reduce == 'mean':
            num = end - start
            real_records = {key: 0.0 for key in records[0].keys()}
            for key in real_records.keys():
                for record in records:
                    real_records[key] += record[key]
            
            for key, value in real_records.items():
                real_records[key] = value / num
        else:
            real_records = records
        
        return real_records


def attach_timer(engine: ig_engine.Engine, engine_name: str,
                 progress_bar: Optional[ig_c_handlers.ProgressBar] = None):
    """
    Use the Timer to measure average time it takes to process a single batch of samples.

    :param engine: The engine to attach timer to.
    :param engine_name: The name of engine.
    :param progress_bar: The progress bar.
    """
    timer = ig_handlers.Timer(average = True)
    timer.attach(
            engine,
            start = ig_engine.Events.EPOCH_STARTED,
            resume = ig_engine.Events.ITERATION_STARTED,
            pause = ig_engine.Events.ITERATION_COMPLETED,
            step = ig_engine.Events.ITERATION_COMPLETED)
    
    timer_total = ig_handlers.Timer()
    timer_total.attach(
            engine,
            start = ig_engine.Events.EPOCH_STARTED,
            pause = ig_engine.Events.EPOCH_COMPLETED)
    
    @engine.on(ig_engine.Events.EPOCH_COMPLETED)
    def print_times(_engine: ig_engine.Engine):
        """
        Print time per batch.

        :param _engine: The engine
        """
        if _engine.state.max_epochs == 1:
            message = "time per batch is {:.4f}s, total time is {:.4f}s.".format(
                    timer.value(), timer_total.value())
        else:
            message = "epoch [{}/{}], time per batch is {:.4f}s, total time is {:.4f}s.".format(
                    _engine.state.epoch, _engine.state.max_epochs,
                    timer.value(), timer_total.value())
        if progress_bar:
            progress_bar.log_message(" ".join([engine_name.capitalize(), message]))
        logging.getLogger(engine_name).info(message.capitalize())
        timer.reset()
