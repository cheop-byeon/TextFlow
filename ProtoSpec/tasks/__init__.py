import inspect
from pprint import pprint

from . import (ids_auto_complete, ids_startup, ids_edit)

TASK_REGISTRY = {
    "ids_auto_complete": ids_auto_complete.IdsTextFlow,
    "ids_startup": ids_startup.IdsTextFlow,
    "ids_edit": ids_edit.IdsTextFlow,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "dataset_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            if args and hasattr(args, 'load_dataset_path') and args.load_dataset_path:
                kwargs["dataset_path"] = args.load_dataset_path
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
