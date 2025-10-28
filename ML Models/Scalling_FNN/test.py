
class Animal :
    def eat(self):
        print("nom nom")

class Dog(Animal) : 
    def __init__(self, name):
        self.name = name
    def bark(self):
        print(f"{self.name} says woof !")

d1 = Dog("Luna")
d1.bark()
d1.eat()

d2 = Dog("Leo")
d2.bark()   

d3 = Dog("seki")
d3.bark()


