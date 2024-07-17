# SBB NER with Spacy

A simple spaCy pipeline to perform NER using a [sbb_ner](https://github.com/qurator-spk/sbb_ner) server.

## Example:

```python
import spacy
import sbb_ner_with_spacy

text = "Major Zeerleder, gutsbesizzer zu Steinegg in Thurgau, der\n" \
    "seit mer als 20. iaren, unter meinem dache das gastrecht genoss,\n" \
    "kam vor 3. wochen zu mir auf besuch."

nlp = spacy.load("de_core_news_sm", disable = ['ner'])
nlp.add_pipe("sbb_ner", config={"api_ner": "http://127.0.0.1:5000"})
doc = nlp(text)

spacy.displacy.serve(doc, style="ent")
```
