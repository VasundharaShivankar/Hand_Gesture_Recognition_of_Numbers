import pickle

with open('extracted_landmarks.pickle', 'rb') as f:
    data = pickle.load(f)

print('Type:', type(data))
print('Keys:', list(data.keys()) if isinstance(data, dict) else 'Not a dict')
if isinstance(data, dict):
    for key, value in data.items():
        print(f'Key: {key}, Type: {type(value)}, Length: {len(value)}')
        if value:
            print(f'First item in {key}: {value[0]}')
else:
    print('Length:', len(data))
    for i, item in enumerate(data):
        print(f'Item {i}: label={item[1]}, landmarks_len={len(item[0])}')
