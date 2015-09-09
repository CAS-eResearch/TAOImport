import os, importlib, inspect
from Module import Module

def find_modules():
    modules = []
    dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
    for entry in os.listdir(dir):
        path = os.path.join(dir, entry)
        if os.path.isfile(path) and entry[-3:].lower() == '.py':
            modules.extend(import_modules(entry[:-3]))
    return list(set(modules))

def import_modules(entry):
    modules = []
    mod = importlib.import_module('tao.' + entry)
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and obj != Module and issubclass(obj, Module):
            modules.append(obj)
    return modules
