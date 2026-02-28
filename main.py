"""
Smart Attendance System using Face Recognition
Main CLI Interface for the Application

This script provides a simple command-line menu to access all features:
1. Register New Student
2. Train Model
3. Start Attendance
4. Exit
"""

import os
import sys
import csv

# Always work from the directory where this script lives,
# so that relative paths (dataset/, trainer/, attendance/) resolve correctly
# even when VS Code or the terminal starts in a different folder.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

from register_face import register_student
from train_model import train_model
from recognize_attendance import start_attendance


def clear_screen():
    """Clear console screen for better UI"""
    os.system('cls' if os.name == 'nt' else 'clear')


def display_menu():
    """Display the main menu"""
    print("\n" + "="*60)
    print("     SMART ATTENDANCE SYSTEM USING FACE RECOGNITION")
    print("="*60)
    print("\n1. Register New Student")
    print("2. Train Face Recognition Model")
    print("3. Start Attendance System")
    print("4. View Registered Students")
    print("5. View Attendance Records")
    print("6. Exit")
    print("\n" + "-"*60)


def view_registered_students():
    """Display all registered students"""
    dataset_path = os.path.join(_SCRIPT_DIR, 'dataset')
    
    if not os.path.exists(dataset_path):
        print("\n❌ No students registered yet!")
        return
    
    students = [folder for folder in os.listdir(dataset_path) 
                if os.path.isdir(os.path.join(dataset_path, folder))]
    
    if not students:
        print("\n❌ No students registered yet!")
        return
    
    print("\n" + "-"*60)
    print("[*] REGISTERED STUDENTS")
    print("-"*60)
    
    for student_id in sorted(students):
        student_path = os.path.join(dataset_path, student_id)
        info_file = os.path.join(student_path, 'info.txt')
        
        student_name = student_id
        sample_count = 0
        
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    for line in f:
                        if line.startswith('Name:'):
                            student_name = line.split(':', 1)[1].strip()
                            break
            except:
                pass
        
        # Count face samples
        for file in os.listdir(student_path):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                sample_count += 1
        
        print(f"\n  ID: {student_id}")
        print(f"  Name: {student_name}")
        print(f"  Samples: {sample_count}")
    
    print("\n" + "-"*60)
    input("\nPress Enter to continue...")


def view_attendance_records():
    """Display attendance records"""
    attendance_file = os.path.join(_SCRIPT_DIR, 'attendance', 'attendance.csv')
    
    if not os.path.exists(attendance_file):
        print("\n❌ No attendance records found!")
        return
    
    print("\n" + "-"*60)
    print("📊 ATTENDANCE RECORDS")
    print("-"*60 + "\n")
    
    try:
        with open(attendance_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            
            if not headers:
                print("❌ Attendance file is empty!")
                return
            
            # Print header
            print(f"{'Student ID':<12} {'Name':<20} {'Date':<12} {'Time':<10}")
            print("-"*60)
            
            # Print records
            row_count = 0
            for row in reader:
                if len(row) >= 4:
                    print(f"{row[0]:<12} {row[1]:<20} {row[2]:<12} {row[3]:<10}")
                    row_count += 1
            
            if row_count == 0:
                print("(No attendance records)")
            else:
                print("-"*60)
                print(f"Total Records: {row_count}")
    
    except Exception as e:
        print(f"❌ Error reading attendance: {str(e)}")
    
    print("\n" + "-"*60)
    input("\nPress Enter to continue...")


def main():
    """Main function to run the CLI menu"""
    while True:
        clear_screen()
        display_menu()
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            clear_screen()
            print("\n" + "="*60)
            print("     STUDENT REGISTRATION")
            print("="*60)
            register_student()
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            clear_screen()
            print("\n" + "="*60)
            print("     TRAINING FACE RECOGNITION MODEL")
            print("="*60)
            train_model()
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            clear_screen()
            print("\n" + "="*60)
            print("     STARTING ATTENDANCE SYSTEM")
            print("="*60)
            start_attendance()
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            clear_screen()
            view_registered_students()
            
        elif choice == '5':
            clear_screen()
            view_attendance_records()
            
        elif choice == '6':
            print("\n\nThank you for using Smart Attendance System!")
            print("Goodbye!\n")
            sys.exit(0)
            
        else:
            print("\n❌ Invalid choice! Please enter 1-6.")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
