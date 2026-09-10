"""A multinomial Naive Bayes intent model trained entirely from local examples."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
INTENTS_PATH = BASE / 'intents.json'
MODEL_PATH = BASE / 'data' / 'intent_model.json'


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.casefold())


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class IntentModel:
    def __init__(self, data):
        self.data = data
        self.vocabulary = set(data['vocabulary'])

    @classmethod
    def train(cls, intents):
        counts, vocabulary, seen = {}, set(), {}
        for intent in intents:
            tag = intent['tag']
            if tag in counts or not intent['patterns']:
                raise ValueError('Each intent needs a unique tag and training examples.')
            counts[tag] = Counter()
            for pattern in intent['patterns']:
                words = tokens(pattern)
                if not words:
                    raise ValueError('Training examples cannot be empty.')
                key = tuple(words)
                if key in seen and seen[key] != tag:
                    raise ValueError(f'Conflicting training example: {pattern}')
                seen[key] = tag
                counts[tag].update(words)
                vocabulary.update(words)
        if len(counts) < 2:
            raise ValueError('At least two intents are required.')
        weights = {}
        for tag, count in counts.items():
            denominator = sum(count.values()) + len(vocabulary)
            weights[tag] = {word: math.log((count[word] + 1) / denominator)
                            for word in sorted(vocabulary)}
        return cls({'version': 1, 'vocabulary': sorted(vocabulary), 'weights': weights})

    def predict(self, text):
        words = tokens(text)
        known = [word for word in words if word in self.vocabulary]
        if not known or len(known) / max(1, len(words)) < 0.5:
            return None, 0.0
        scores = {tag: sum(weight[word] for word in known)
                  for tag, weight in self.data['weights'].items()}
        maximum = max(scores.values())
        probabilities = {tag: math.exp(score - maximum) for tag, score in scores.items()}
        total = sum(probabilities.values())
        ranked = sorted(((p / total, tag) for tag, p in probabilities.items()), reverse=True)
        confidence, tag = ranked[0]
        if confidence < 0.55 or confidence - ranked[1][0] < 0.15:
            return None, confidence
        return tag, confidence


def train_and_save(source=INTENTS_PATH, destination=MODEL_PATH):
    source, destination = Path(source), Path(destination)
    dataset = json.loads(source.read_text(encoding='utf-8'))
    model = IntentModel.train(dataset['intents'])
    model.data['source_hash'] = fingerprint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(model.data, indent=2), encoding='utf-8')
    temporary.replace(destination)
    return model


def load_model():
    try:
        data = json.loads(MODEL_PATH.read_text(encoding='utf-8'))
        if data.get('version') == 1 and data.get('source_hash') == fingerprint(INTENTS_PATH):
            return IntentModel(data)
    except (OSError, ValueError, KeyError):
        pass
    return train_and_save()
