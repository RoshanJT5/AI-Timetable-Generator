"""
Normalization utility for program and branch keys.

This ensures consistent storage format across the entire system.
All program and branch values MUST be normalized before storage.
"""

def normalize_key(value: str) -> str:
    """
    Normalize program/branch strings to canonical format.
    
    Rules:
    - Convert to lowercase
    - Remove dots (.)
    - Remove spaces
    - Strip whitespace
    
    Examples:
        "B.Tech" -> "btech"
        "B tech" -> "btech"
        "Computer Science" -> "computerscience"
        "MBA" -> "mba"
    
    Args:
        value: Raw string input from user/import
        
    Returns:
        Normalized string for database storage
    """
    if not value:
        return None
    return value.strip().lower().replace(".", "").replace(" ", "")


def normalize_program_branch(program: str, branch: str) -> tuple:
    """
    Normalize both program and branch at once.
    
    Args:
        program: Raw program string
        branch: Raw branch string
        
    Returns:
        Tuple of (normalized_program, normalized_branch)
    """
    return (normalize_key(program), normalize_key(branch))
