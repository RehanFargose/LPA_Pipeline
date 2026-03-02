# import os
# from dotenv import load_dotenv
# from spacy_llm.util import assemble_from_config
# from confection import Config

# load_dotenv()

# raw_config = {
#     "nlp": {"lang": "en", "pipeline": ["llm"]},
#     "components": {
#         "llm": {
#             "factory": "llm",  # <--- THIS IS THE FIX: Explicitly name the factory
#             "task": {
#                 "@llm_tasks": "spacy.NER.v2",
#                 "labels": ["PETITIONER", "RESPONDENT", "COURT", "STATUTE", "AMOUNT", "DATE"],
#                 "label_definitions": {
#                     "PETITIONER": "The party bringing the appeal or petition.",
#                     "RESPONDENT": "The party against whom the appeal is filed.",
#                     "COURT": "The legal body or bench hearing the case.",
#                     "STATUTE": "Legal acts, sections, or articles cited.",
#                     "AMOUNT": "Specific monetary figures mentioned.",
#                     "DATE": "Specific dates mentioned in the proceedings."
#                 }
#             },
#             "model": {
#                 "@llm_models": "spacy.Gemini.v1",
#                 "config": {"model_name": "gemini-1.5-pro", "temperature": 0.0}
#             }
#         }
#     }
# }

# config = Config(raw_config)
# nlp = assemble_from_config(config)


import os
from dotenv import load_dotenv
from spacy_llm.util import assemble_from_config
from confection import Config

load_dotenv()

raw_config = {
    "nlp": {"lang": "en", "pipeline": ["llm"]},
    "components": {
        "llm": {
            "factory": "llm",
            "task": {
                "@llm_tasks": "spacy.NER.v2",
                "labels": ["PETITIONER", "RESPONDENT", "COURT", "STATUTE", "AMOUNT", "DATE"],
                "label_definitions": {
                    "PETITIONER": "The party bringing the appeal or petition.",
                    "RESPONDENT": "The party against whom the appeal is filed.",
                    "COURT": "The legal body or bench hearing the case.",
                    "STATUTE": "Legal acts, sections, or articles cited.",
                    "AMOUNT": "Specific monetary figures mentioned.",
                    "DATE": "Specific dates mentioned in the proceedings."
                }
            },
            "model": {
                "@llm_models": "spacy.PaLM.v2",
                "name": "text-bison-001",  # Changed from chat-bison-001
                "config": {"temperature": 0.0}
            }
        }
    }
}

config = Config(raw_config)
nlp = assemble_from_config(config)