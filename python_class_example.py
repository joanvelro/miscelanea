class Car:
    def __init__(self):
        self.speed = 100
        self.odometer = 0
        self.time = 1

    def say_state(self):
        print("I'm going {} km/h!".format(self.speed))

    # increase the current speed 5 km/h
    def accelerate(self):
        self.speed += 5

    # reduce the current speed 5 km/h
    def brake(self):
        self.speed -= 5

    # Increment counter of the odometer based on the current speed and the time
    def step(self):
        self.odometer += self.speed
        self.time += 1

    # calculate average spped
    def average_speed(self):
        return self.odometer / self.time


if __name__ == '__main__':
    my_car = Car()
    print("I'm a car!")
    my_car.say_state()
    while True:
        print('\n')
        action = input("What should I do? [A]ccelerate, [B]rake, "
                       "show [L]ecture, or show average [S]peed?").upper()
        if action not in "ABOS" or len(action) != 1:
            print("I don't know how to do that")
            continue
        if action == 'A':
            my_car.accelerate()
        if action == 'B':
            my_car.brake()
        if action == 'L':
            print("The car has driven {} kilometers".format(my_car.odometer))
        if action == 'S':
            print('\n')
            print("The car's average speed was {} kph".format(my_car.average_speed()))
        my_car.step()
        my_car.say_state()
