# vital-llm-reasoner

# Ensemble Reasoning

Implementation to address deployment in:
Llama.cpp: CPU and GPU
vLLM: GPU
transformers (huggingface): GPU, baseline for ensemble attention implementation

# Version 1 Tags

<|ensemble:member|>
<|/ensemble:member|>

<|ensemble_result:member|>
<|/ensemble_result:member|>

member is one of:
web_search
wikidata_search
kgraph_search
kgraph_traverse
logic_query
code_executor

logic_query terms for testing:
friend(?Friend)
search_friends('search term', ?Friend)
get_friend('friend_uri', ?Friend)
traverse('uri', ?Node)
traverse_incoming('uri', ?Node)
traverse_outgoing('uri', ?Node)

# Version 2 Tags
# Use JSON Schema
 include request id
 