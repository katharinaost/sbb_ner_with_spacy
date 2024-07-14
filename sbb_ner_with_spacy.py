import sys
import requests
import json
from itertools import tee, islice, chain
import spacy
from spacy.language import Language
from spacy.tokens import Doc

def previous_and_next(some_iterable):
    # https://stackoverflow.com/questions/1011938/loop-that-also-accesses-previous-and-next-values
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)

@Language.factory("sbb_ned", default_config={"api_ner": None, "model_id": 1})
def sbb_ned(nlp: Language, name: str, api_ner: str, model_id: int):
    return SBBNedComponent(nlp, api_ner, model_id)

class SBBNedComponent:
    def __init__(self, nlp: Language, api_ner: str, model_id: int):
        self.api_ner = api_ner
        self.model_id = model_id
        
    def make_web_request(self, request_url: str, request_data: str):
        request_headers={'Content-Type': 'application/json'}
        try:
            response = requests.post(request_url, data=request_data, headers=request_headers)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)
        return response
        
    def __call__(self, doc: Doc) -> Doc:
        spans = []
        
        request_text={'text': doc.text}
        response = self.make_web_request(self.api_ner+"/ner/"+str(self.model_id), json.dumps(request_text))
        response_json=response.json()

        character_count=0
        start_character=0
        end_character=0
        
        for sentence in response_json:
            for previous_word, current_word, next_word in previous_and_next(sentence):
                temp_word=""
                i = 0
                empty_chars = 0
                while (len(temp_word) < len(current_word["word"])) and (character_count+i+1 < len(doc.text)):
                    # whitespaces are stripped from the ner output, count how many there were in the original
                    # to align ner tags properly
                    if doc.text[character_count+i:character_count+i+1] not in [' ', '\n', '\r', '\t']:
                        temp_word += doc.text[character_count+i:character_count+i+1]
                    else:
                        empty_chars += 1
                    i += 1
                if not current_word["prediction"].startswith("O"):
                    if current_word["prediction"].startswith("B"):
                        # beginning of new ner tag
                        start_character=character_count+empty_chars
                    if current_word["prediction"].startswith("I") and (not previous_word or previous_word["prediction"].startswith("O")):
                        # sometimes a new ner tag starts with I, not B
                        start_character=character_count+empty_chars
                    if next_word["prediction"].startswith("B") or next_word["prediction"].startswith("O"):
                        # end of current ner tag
                        end_character=character_count+empty_chars+len(current_word["word"])
                        spans.append(doc.char_span(start_character, end_character, label=current_word["prediction"][2:]))
                character_count += i
        doc.ents = spans
        return doc

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        text_de = f.read()
except OSError:
    raise SystemExit('Could not open/read file: %s' % sys.argv[1])

nlp_model_de = spacy.load("de_core_news_sm", disable = ['ner'])
nlp_model_de.add_pipe("sbb_ned", config={"api_ner": "http://127.0.0.1:5000"})
doc_de = nlp_model_de(text_de)

spacy.displacy.serve(doc_de, style="ent")