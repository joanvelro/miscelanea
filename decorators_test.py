
# https://medium.com/swlh/handling-exceptions-in-python-a-cleaner-way-using-decorators-fae22aa0abec

def exception_handler(func):
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        # except TypeError:
        #    print(f"{func.__name__} only takes numbers as the argument")
        except Exception as exception_msg:
            print('(!) Error in {}: {} '.format(func.__name__, str(exception_msg)))

    return inner_function


@exception_handler
def area_square(length):
    print(length * length)


@exception_handler
def area_circle(radius):
    print(3.14 * radius * radius)


@exception_handler
def area_rectangle(length, breadth):
    print(length * breadth)


area_square(2)
area_circle(2)
area_rectangle(2, 4)
area_square("some_str")
area_circle("some_other_str")
area_rectangle("some_other_rectangle")
