import medialpy

def get_medical_abbrieviation_meaning(
    term: str,
) -> str:
    """
    Get the meaning of a medical abbreviation.
    
    Args:
        term(str): The medical abbreviation or term name to look up.
    Returns:
        str: The meaning of the medical abbreviation.
    """
    
    try:
        import medialpy

        term = medialpy.find(term
                            )
        if term is not None:
            result = f"""Abbreviation: {term.abbreviation}
Meaning: {term.meaning}"""
            return result
        else:
            return f"Abbreviation '{term}' not found." 
    except ImportError:
        return "medialpy is not installed. Please install it to use this function." 
    