import random
import string

def generate_random_string(length):
    """
    Generates a random string of characters with a specified length.

    Args:
        length (int): The length of the random string to generate.

    Returns:
        str: The randomly generated string.
    """
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string
