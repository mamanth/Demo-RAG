from transformers import pipeline

class ContradictionDetector:
    def __init__(self, model_name='roberta-large-mnli'):
        # note: large model; if you have limited resources, replace with a smaller MNLI model
        self.nli = pipeline('text-classification', model=model_name)

    def detect(self, reference_text, others):
        out = []
        for i, o in enumerate(others):
            pair = reference_text + ' </s></s> ' + o['text']
            res = self.nli(pair)
            if isinstance(res, list) and res:
                out.append({'index':i, 'label':res[0]['label'], 'score':float(res[0].get('score',0.0))})
        return out
