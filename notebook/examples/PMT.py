# %%
# !curl "https://uts-ws.nlm.nih.gov/download?url=https://download.nlm.nih.gov/umls/kss/2024AB/umls-2024AB-metathesaurus-full.zip&apiKey=[YOUR_API_KEY]" -o db_cache/umls-2024AB-metathesaurus-full.zip

# %%
from owlready2 import *
from owlready2.pymedtermino2 import *
from owlready2.pymedtermino2.umls import *
default_world.set_backend(filename = "db_cache/pym2.sqlite3",
                              exclusive = False,
                              enable_thread_parallelism = True)
import_umls("db_cache/umls-2024AB-metathesaurus-full.zip", terminologies = ["ICD10", "SNOMEDCT_US", "CUI"])
default_world.save()

# %%
# from owlready2 import *
# default_world.set_backend(filename = "pym.sqlite3")
PYM = get_ontology("http://PYM/").load()
SNOMEDCT_US = PYM["SNOMEDCT_US"]

# %%
concept = SNOMEDCT_US[302509004]
concept

# %%
query = "Alzheimer"

se


type(search_result)



# %%
search_result[0].groups


# %%
import json


from smolagents import tool

@tool
def PyMedTermino_search(query:str, top_k:str=5) -> str:
    """
    Search from PyMedTermino about relevant medical concepts and relations. Collate and parse the search result into a structured dictionary and convert it to a string.

    Args:
        query(str): The first item from the search result (e.g., search_result[0]).
        top_k(int): The number of top results to return.
    Returns:
        str: A JSON-like string containing the label, synonyms, parents, and children of the search result item.
    """
    search_result = SNOMEDCT_US.search(query)
    
    if len(search_result) == 0:
        return "[No results found.]"
    
    else:
        results = []
        for search_result_item in search_result[:top_k]:
            
            result_dict = {
                "label": search_result_item.label,
                "synonyms": search_result_item.synonyms,
                "parents": [parent.label for parent in search_result_item.parents],
                "children": [child.label for child in search_result_item.children],
                "groups": [group.label for group in search_result_item.groups],
            }
            
            results.append(result_dict)
    return json.dumps(results, indent=4)

# Example usage
parsed_result_str = collate_search_result('Alzheimer')
print(parsed_result_str)

# %%



