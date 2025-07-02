import json
import os


student = {}
filename = "C:\\Users\\Nirav\\OneDrive\\Desktop\\Learning\\MachineLearning\\students.json"

def load_data():
    global student
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as file:
                student = json.load(file)
        except Exception as e:
            print(" Error in data loading ")
            student = {}
    return student

def save_data():
    with open(filename, 'w') as file:
        json.dump(student,file, indent=4)

def calculate_gpa(grades):
    if not grades:
        return 0.0
    gpa_points = []
    for grade in grades:
        if grade >= 90:
            gpa_points.append(4.0)
        elif grade >= 80:
            gpa_points.append(3.0)
        elif grade >= 70:
            gpa_points.append(2.0)
        elif grade >= 60:
            gpa_points.append(1.0)
        else:
            gpa_points.append(0.0)
    return round(sum(gpa_points) / len(gpa_points), 2)

def add_student():
    name = input("Enter the student name: ").strip().title()
    if name in student:
        print("Student already Exists")
        return 
    grades =[]
    while True:
        try:
            grade = input("Enter grade between 0-100 or done:")
            if grade.lower() == 'done':
                break
            grade = float(grade)
            if 0<= grade <= 100:
                grades.append(grade)
            else:
                print("Grade must be between 0 and 100.")
        except ValueError:
            print(ValueError)
    student[name] = grades
    save_data()
    print(f"Added {name} with grades {grades}")


def edit_student():
    name = input("Enter the name of the student you want to edit: ").strip().title()
    if name not in student:
        print("Student does not exist")
        return
    grades =[]
    while True:
        try:
            grade = input("Enter grade between 0-100 or done:")
            if grade.lower() == 'done':
                break
            grade = float(grade)
            if 0<= grade <= 100:
                grades.append(grade)
            else:
                print("Grade must be between 0 and 100.")
        except ValueError:
            print("Input Correct value")
    student[name] = grades
    save_data()
    print(f"Edited {name} with grades {grades}")

def remove_student():
    name = input("Enter the name of the student you want to remove4: ").strip().title()
    if name not in student:
        print("Student does not exist")
        return

    del student[name]
    save_data()
    print("Deleted.....")

def display_student():
    if not student:
        print("No student in the database")
        return
    print("\nStudent Grade Report:")
    print("-" * 30)
    for name, grades in student.items():
        gpa = calculate_gpa(grades)
        print(f"Name: {name}, Grades: {grades}, GPA: {gpa}")
    print("-" * 30)
        
def main():
    load_data()
    while True:
        print("\nStudent Grade Management System")
        print("1. Add student")
        print("2. Edit student grades")
        print("3. Remove student")
        print("4. Display all students")
        print("5. Exit")
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            add_student()
        elif choice == '2':
            edit_student()
        elif choice == '3':
            remove_student()
        elif choice == '4':
            display_student()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()