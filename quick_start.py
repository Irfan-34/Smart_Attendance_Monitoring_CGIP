#!/usr/bin/env python3
"""
Quick Start Script - Validates system setup and runs tests
"""

import os
import sys
import cv2
import numpy as np

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step_num, text):
    print(f"\n[{step_num}] {text}")

def check_mark():
    return "[OK]"

def cross_mark():
    return "[X]"

def test_system():
    """Run comprehensive system tests"""
    
    print_header("SMART ATTENDANCE SYSTEM - SETUP VALIDATION")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Python version
    print_step(1, "Checking Python version...")
    tests_total += 1
    if sys.version_info >= (3, 7):
        print(f"   {check_mark()} Python {sys.version.split()[0]} (OK)")
        tests_passed += 1
    else:
        print(f"   {cross_mark()} Python {sys.version.split()[0]} (Need 3.7+)")
    
    # Test 2: OpenCV
    print_step(2, "Checking OpenCV installation...")
    tests_total += 1
    try:
        print(f"   {check_mark()} OpenCV {cv2.__version__} installed")
        tests_passed += 1
    except:
        print(f"   {cross_mark()} OpenCV not found")
        print("   Run: pip install opencv-python opencv-contrib-python")
    
    # Test 3: LBPH Face Recognizer
    print_step(3, "Checking LBPH Face Recognizer...")
    tests_total += 1
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print(f"   {check_mark()} LBPH Face Recognizer available")
        tests_passed += 1
    except:
        print(f"   {cross_mark()} LBPH module not found")
        print("   Run: pip install opencv-contrib-python")
    
    # Test 4: Webcam
    print_step(4, "Checking webcam access...")
    tests_total += 1
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"   {check_mark()} Webcam working ({frame.shape[1]}x{frame.shape[0]}px)")
            tests_passed += 1
        else:
            print(f"   {cross_mark()} Webcam detected but not readable")
        cap.release()
    else:
        print(f"   {cross_mark()} Webcam not found")
    
    # Test 5: Directory structure
    print_step(5, "Checking directory structure...")
    tests_total += 1
    dirs = ['dataset', 'trainer', 'attendance']
    missing = []
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            missing.append(d)
    
    if not missing:
        print(f"   {check_mark()} All required directories exist")
    else:
        print(f"   [*] Created missing directories: {', '.join(missing)}")
    tests_passed += 1
    
    # Test 6: Registered students
    print_step(6, "Checking registered students...")
    tests_total += 1
    students = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
    if students:
        print(f"   {check_mark()} Found {len(students)} registered students:")
        for student in sorted(students):
            samples = len([f for f in os.listdir(f'dataset/{student}') if f.endswith(('.jpg', '.png'))])
            status = "[OK]" if samples >= 20 else "[!] " if samples >= 10 else "[X]"
            print(f"       {status} {student}: {samples} samples")
        tests_passed += 1
    else:
        print(f"   [!]  No students registered yet")
        tests_passed += 1
    
    # Test 7: Trained model
    print_step(7, "Checking trained model...")
    tests_total += 1
    if os.path.exists('trainer/trainer.yml'):
        print(f"   {check_mark()} Model found (trainer.yml)")
        if os.path.exists('trainer/labels.txt'):
            with open('trainer/labels.txt', 'r') as f:
                labels = len(f.readlines())
            print(f"   {check_mark()} Labels found ({labels} students)")
        tests_passed += 1
    else:
        print(f"   [!]  Model not trained yet (need to run Option 2)")
        tests_passed += 1
    
    # Test 8: Attendance file
    print_step(8, "Checking attendance records...")
    tests_total += 1
    if os.path.exists('attendance/attendance.csv'):
        with open('attendance/attendance.csv', 'r') as f:
            lines = len(f.readlines())
        print(f"   {check_mark()} Attendance file found ({lines} records)")
    else:
        print(f"   [i]  Attendance file will be created on first mark")
    tests_passed += 1
    
    # Results
    print_header("VALIDATION RESULTS")
    print(f"\nTests Passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print(f"\n{check_mark()}{check_mark()} System is READY! [OK][OK]")
        print("\nNext steps:")
        print("  1. Run: python main.py")
        print("  2. Choose Option 1 to register students")
        print("  3. Choose Option 2 to train model")
        print("  4. Choose Option 3 for attendance")
        return True
    else:
        print(f"\n{cross_mark()} Please fix issues above before proceeding")
        print("\nSee SETUP_GUIDE.md for troubleshooting")
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
