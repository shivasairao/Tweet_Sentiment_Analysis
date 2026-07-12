import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Quietly download necessary assets
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

def clean_tweet(text):
    """ Cleans raw text for semantic parsing. """
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    # Strip URL structures, twitter handles (@), and formatting symbols
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    
    tokens = word_tokenize(text)
    # Maintain semantic density: remove stopwords, drop numbers/single characters
    cleaned_tokens = [w for w in tokens if w not in STOP_WORDS and w.isalpha() and len(w) > 1]
    return cleaned_tokens