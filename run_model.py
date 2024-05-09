import pathlib
import time
from typing import List, Tuple

import torch

import configs
from train import FineTuneProcess, Process
from utils import CsvSaver


def run_model_with_seeds(name: str, network, conf: configs.Configuration, batch_size: int, epochs: int,
                         device: torch.device, worker_num: int, additional_labels: bool, output_dir: pathlib.Path,
                         room_seed_pairs: List[Tuple[int, int]]):
    """
    
    :param name:
    :param network:
    :param conf:
    :param batch_size:
    :param epochs:
    :param device:
    :param worker_num:
    :param additional_labels:
    :param output_dir:
    :param room_seed_pairs:
    """
    with CsvSaver((output_dir / 'result.csv')) as final_result_saver, \
            CsvSaver((output_dir / 'result_best.csv')) as best_result_saver, \
            CsvSaver((output_dir / 'time.csv')) as time_saver:
        for room, seed in room_seed_pairs:
            print("room {} with seed {}".format(room, seed))
            final_result = {'room': room, 'seed': seed}
            best_result = {'room': room, 'seed': seed}
            
            start = time.time()
            result = run_full(
                    network = network,
                    configuration = conf,
                    batch_size = batch_size,
                    epochs = epochs,
                    target_room = room,
                    device = device,
                    worker_num = worker_num,
                    seed = seed,
                    output_prefix = f'model_{name}_room{room}_seed{seed}',
                    print_model_to_console = False,
                    additional_labels = additional_labels)
            end = time.time()
            
            print("room {} with seed {} costs {} seconds, results are {}".format(room, seed, end - start, result))
            time_saver.write_row({'room': room, 'seed': seed, 'time': end - start})
            final_result.update(result['final'])
            final_result_saver.write_row(final_result)
            best_result.update(result['best'])
            best_result_saver.write_row(best_result)
        
        print(final_result_saver.data)
        print(best_result_saver.data)


def run_full(network, configuration: configs.Configuration, batch_size: int, epochs: int,
             target_room: int,
             device: torch.device, worker_num: int,
             seed: int, output_prefix: str, print_model_to_console: bool,
             additional_labels: bool) -> dict:
    """
    Run full.

    :param network:
    :param configuration:
    :param batch_size:
    :param target_room:
    :param device:
    :param epochs:
    :param worker_num:
    :param seed:
    :param output_prefix:
    :param print_model_to_console:
    :param additional_labels:
    :return:
    """
    process = Process(configuration, device, network, seed, output_prefix = output_prefix)
    process.load_data(batch_size, worker_num = worker_num, target_room = target_room,
                      additional_labels = additional_labels)
    process.print_model(print_model_to_console)
    process.setup_trainer()
    process.setup_tester()
    process.train(epochs)
    process.save_model('final_model.pt')
    result = dict()
    process.test()
    result['final'] = process.get_test_result()
    process.test(process.best_model_path)
    result['best'] = process.get_test_result()
    process.quit()
    return result


def run_finetune_full(network, configuration: configs.Configuration, batch_size: int, epochs: int,
                      target_room: int,
                      saved_model_path: pathlib.Path,
                      device: torch.device, worker_num: int,
                      seed: int, output_prefix: str, print_model_to_console: bool,
                      additional_labels: bool) -> dict:
    """
    Run full.

    :param network:
    :param configuration:
    :param batch_size:
    :param target_room:
    :param device:
    :param epochs:
    :param worker_num:
    :param saved_model_path:
    :param seed:
    :param output_prefix:
    :param print_model_to_console:
    :param additional_labels:
    :return:
    """
    process = FineTuneProcess(configuration, device, network, seed, output_prefix = output_prefix)
    process.load_data(batch_size, worker_num = worker_num, target_room = target_room,
                      additional_labels = additional_labels)
    process.print_model(print_model_to_console)
    process.setup_trainer()
    process.setup_tester()
    process.load_model(saved_model_path)
    process.train(epochs)
    process.save_model('final_model.pt')
    result = dict()
    process.test()
    result['final'] = process.get_test_result()
    process.test(process.best_model_path)
    result['best'] = process.get_test_result()
    process.quit()
    return result
