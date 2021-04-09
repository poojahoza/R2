import json
import requests

MY_GCUBE_TOKEN = '4f3275bc-555a-4627-99b0-36e1fd7ef45f-843339462'


class WATAnnotation:

    def __init__(self,d):
        self.start = d['start']
        self.end = d['end']
        self.rho = d['rho']
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']
        self.spot = d['spot']
        self.wiki_id = d['id']
        self.wiki_title = d['title'].replace("_"," ")

    def json_dict(self):

        return{
            'wiki_title': self.wiki_title,
            'wiki_id': self.wiki_id,
            'start': self.start,
            'end': self.end,
            'rho': self.rho,
            'spot': self.spot
        }


def wat_entity_linking(text):

    wat_url='https://wat.d4science.org/wat/tag/tag'
    payload=[("gcube-token", MY_GCUBE_TOKEN),
             ("text", text),
             ("lang", "en"),
             ("tokenizer", "nlp4j"),
             ("debug", 9),
             ("method", "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")
    ]

    response = requests.get(wat_url, params=payload)
    #print(response)
    return [WATAnnotation(a) for a in response.json()['annotations'] if a['rho'] >= 0.1]
"""
def print_wat_annotations(wat_annotations):
    json_list = [w.json_dict() for w in wat_annotations]
    print(json.dumps(json_list,indent=4))
wat_annotations = wat_entity_linking("Per the The ICU Book The first rule of antibiotics is try not to use them, and the second rule is try not to use too many of them. Inappropriate antibiotic treatment and overuse of antibiotics have contributed to the emergence of antibiotic-resistant bacteria. Self prescription of antibiotics is an example of misuse. Many antibiotics are frequently prescribed to treat symptoms or diseases that do not respond to antibiotics or that are likely to resolve without treatment. Also, incorrect or suboptimal antibiotics are prescribed for certain bacterial infections. The overuse of antibiotics, like penicillin and erythromycin, has been associated with emerging antibiotic resistance since the 1950s. Widespread usage of antibiotics in hospitals has also been associated with increases in bacterial strains and species that no longer respond to treatment with the most common antibiotics.")
print_wat_annotations(wat_annotations)
"""