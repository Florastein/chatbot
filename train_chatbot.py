"""Train our own intent model from local examples. No downloads or pretrained weights."""
import json
from local_model import INTENTS_PATH, MODEL_PATH, train_and_save

if __name__ == '__main__':
    model = train_and_save()
    dataset = json.loads(INTENTS_PATH.read_text(encoding='utf-8'))
    count = sum(len(i['patterns']) for i in dataset['intents'])
    print(f'Trained {len(model.data["weights"])} intents from {count} examples.')
    print(f'Saved model to {MODEL_PATH}')
