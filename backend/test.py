# backend/test_blip_import.py
try:
    import BLIP_CAM
    print("BLIP_CAM imported successfully!")
except ImportError as e:
    print(f"Error importing BLIP_CAM: {e}")