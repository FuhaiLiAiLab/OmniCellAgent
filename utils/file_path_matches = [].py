file_path_matches = []
search_start = 0
while True:
    idx = potential_content.find('file_path', search_start)
    if idx == -1:
        break
    # Find the first ':' after 'file_path'
    colon_idx = potential_content.find(':', idx)
    if colon_idx == -1:
        break
    # Find the first quote (single or double) after the colon
    quote_idx = -1
    quote_char = ''
    for q in ["'", '"']:
        q_idx = potential_content.find(q, colon_idx + 1)
        if q_idx != -1 and (quote_idx == -1 or q_idx < quote_idx):
            quote_idx = q_idx
            quote_char = q
    if quote_idx == -1:
        search_start = colon_idx + 1
        continue
    # Find the closing quote
    end_quote_idx = potential_content.find(quote_char, quote_idx + 1)
    if end_quote_idx == -1:
        search_start = quote_idx + 1
        continue
    # Extract the file path
    file_path = potential_content[quote_idx + 1:end_quote_idx]
    file_path_matches.append(file_path)
    search_start = end_quote_idx + 1