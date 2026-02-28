"""
Quick test script to diagnose webcam issues
"""
import cv2
import threading
import time

def test_webcam_direct():
    """Test direct webcam access"""
    print("Testing direct webcam access...")
    cap = cv2.VideoCapture(0)
    print(f"  - VideoCapture created: {cap is not None}")
    print(f"  - Is opened: {cap.isOpened()}")
    
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"  - Can read frame: {ret}")
        if ret:
            print(f"  - Frame size: {frame.shape}")
        cap.release()
        print("[OK] Webcam is accessible!")
        return True
    else:
        print("❌ Webcam failed to open")
        cap.release()
        return False

def test_webcam_with_timeout(timeout_seconds=5):
    """Test webcam with timeout"""
    print(f"\nTesting webcam with {timeout_seconds}s timeout...")
    result = [False]
    
    def try_open():
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                result[0] = ret
                cap.release()
        except Exception as e:
            print(f"  Exception: {e}")
    
    thread = threading.Thread(target=try_open, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if result[0]:
        print("[OK] Webcam test passed (with timeout)!")
    else:
        print("❌ Webcam test failed")
    
    return result[0]

def list_available_cameras(max_index=10):
    """Try to find available cameras"""
    print(f"\nScanning for available cameras (checking indices 0-{max_index})...")
    available = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available.append(i)
                print(f"  [OK] Camera at index {i} is available")
            cap.release()
    
    if not available:
        print("  ❌ No cameras found")
    
    return available

if __name__ == "__main__":
    print("="*60)
    print("WEBCAM DIAGNOSTIC TEST")
    print("="*60)
    
    # Test 1: Direct access
    direct_result = test_webcam_direct()
    
    # Test 2: With timeout
    timeout_result = test_webcam_with_timeout(5)
    
    # Test 3: List cameras
    available = list_available_cameras(5)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if available:
        print(f"[OK] Found {len(available)} camera(s): {available}")
    else:
        print("❌ Found NO cameras")
        print("\nPossible solutions:")
        print("  1. Connect a USB webcam or ensure built-in camera is working")
        print("  2. Check Device Manager for camera drivers")
        print("  3. Restart your computer")
        print("  4. Check Windows Settings > Privacy & Security > Camera")
    
    print()
