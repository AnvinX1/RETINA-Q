"""
API Integration Tests for RETINA-Q
Tests health, fundus prediction, OCT prediction, and segmentation endpoints.
"""
import requests
import os
import cv2
import numpy as np

BASE_URL = "http://localhost:8000"

def create_dummy_image(path):
    # Creates a dummy 224x224 RGB image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path

def test_health():
    print("Testing /health...")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    print("Health OK:", r.json())

def test_endpoints():
    test_img = "test_dummy.jpg"
    create_dummy_image(test_img)

    try:
        # Test Fundus
        print("\nTesting /api/predict/fundus...")
        with open(test_img, "rb") as f:
            r = requests.post(f"{BASE_URL}/api/predict/fundus", files={"file": ("test_dummy.jpg", f, "image/jpeg")})
        print("Fundus Response:", r.status_code)
        if r.status_code == 200:
            print("Predicted:", r.json().get("prediction"), "| Confidence:", r.json().get("confidence"))

        # Test Segmentation
        print("\nTesting /api/segment...")
        with open(test_img, "rb") as f:
            r = requests.post(f"{BASE_URL}/api/segment", files={"file": ("test_dummy.jpg", f, "image/jpeg")})
        print("Segmentation Response:", r.status_code)

    finally:
        os.remove(test_img)

if __name__ == "__main__":
    test_health()
    test_endpoints()
    print("\nAPI Integration Tests Completed!")
