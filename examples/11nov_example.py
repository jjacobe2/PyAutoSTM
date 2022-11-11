
class Student():
    def __init__(self, name, major):
        self.name = name
        self.major = major

    def change_name(self, new_name):
        self.name = new_name

def do_greeting(student):
    name = student.name
    major = student.major

    print('Hi ' + name + ' , I am also a ' + major + ' major!')

student1 = Student('John', 'Physics')
print(student1.name)

student1.change_name('Johnny')
do_greeting(student1)