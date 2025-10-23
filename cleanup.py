import os

for i in range(10):
    folder = f'dataset/{i}'
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith('.png'):
                os.remove(os.path.join(folder, f))
                print(f"Removed {os.path.join(folder, f)}")

print("Cleanup complete.")
