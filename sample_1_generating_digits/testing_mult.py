from datetime import datetime as dt
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from itertools import combinations
from multiprocessing import Pool, Manager
from functools import partial

def get_transfer_matrix(n_items):
    all_objectives = np.random.randint(2000, 2200, size=(n_items,))
    tm = np.random.normal(50, 10, size=(n_items, n_items - 1))
    transfer_matrix = []
    for ind, row in enumerate(tm):
        this_row = list(row / sum(row) * np.random.normal(0.8, 0.06))
        this_row = this_row[:ind] + [0] + this_row[ind:]
        transfer_matrix.append(this_row)
    transfer_matrix = np.array(transfer_matrix)
    return all_objectives, transfer_matrix 
 

def compute_objective(items, n_items, all_objectives, transfer_matrix, storage):
    objective = 0
    selection = np.array([1 if item in items else 0 for item in range(n_items)])
    for item in range(n_items):
        if item in items:
            objective += all_objectives[item]
        else:
            objective += all_objectives[item] * np.matmul(selection, transfer_matrix[item])
    return storage.append((items, np.round(objective, 2)))

def main():
    n_items = 100
    all_objectives, transfer_matrix = get_transfer_matrix(n_items)
    manager = Manager()
    managed_list = manager.list()
    partial_func = partial(
        compute_objective,
        n_items=n_items,
        all_objectives=all_objectives,
        transfer_matrix=transfer_matrix,
        storage=managed_list
    )
    all_selections = (
        list(combinations(range(n_items), 25)) +
        list(combinations(range(n_items), 24))
    )

    pool = Pool(processes=4)
    print(dt.now())
    pool.map(partial_func, all_selections)
    final_list = [element for element in managed_list]
    print(dt.now())
    print(len(final_list))


def main2():
    n_items = 100
    all_objectives, transfer_matrix = get_transfer_matrix(n_items)
    manager = Manager()
    managed_list = manager.list()
    partial_func = partial(
        compute_objective,
        n_items=n_items,
        all_objectives=all_objectives,
        transfer_matrix=transfer_matrix,
        storage=managed_list
    )
    all_selections = [
        sorted(np.random.choice(range(100), 80, replace=False))
        for _ in range(100000)
    ]
    
    pool = Pool(processes=8)
    print(dt.now())
    pool.map(partial_func, all_selections)
    final_list = [element for element in managed_list]
    print(dt.now())
    print(len(final_list))


if __name__ == '__main__':
    main2()
