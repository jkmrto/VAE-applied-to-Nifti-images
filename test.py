def print_recursive_dict(dic, file=None, suffix=""):

    for key, item in dic.items():
        if isinstance(item, dict):
            next_suffix = suffix + "{},".format(key)
            print_recursive_dict(dic=item, suffix=next_suffix)
        else:
            print(suffix + "{0}: {1}".format(key, item))

test = {'hola': {'a': 1, 'b': 2}, 'hola3': {'b3': 2, 'a3': 1}}
print_recursive_dict(test)