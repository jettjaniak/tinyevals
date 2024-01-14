def dummy(arg: int | str = None) -> int | str:
    match arg:
        case int(arg):
            return arg + 1
        case str(arg):
            return arg + '1'
        case _:
            return 'dummy'