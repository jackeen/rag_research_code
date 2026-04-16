"""
The sentence fluency experiment.
"""

from langchain_ollama import ChatOllama
import sys_config

llm = ChatOllama(
    base_url=sys_config.OLLAMA_URL_BASE,
    model=sys_config.OLLAMA_GRANITE_MODEL_4_3B_H,
    temperature=0
)

def fluency_score(text: str) -> float:
    prompt = f'''
    You are a language quality evaluator.

    Your task is to assess how fluent and natural a given piece of text is in English.

    Fluency includes:
    - grammatical correctness
    - clarity of expression
    - natural phrasing (native-like)
    - logical flow between sentences

    Instructions:
    - Output a single number between 0 and 1
    - 1 = perfectly fluent and natural
    - 0 = completely broken or incomprehensible
    - Do NOT provide explanations or any extra text
    - Only return the score

    Text to evaluate:

    "{text}"
    '''

    result = llm.invoke(input=prompt).content
    return float(result)


if __name__ == '__main__':
    content = '''
    However, the cold hard truth is that whilst I fundamentally favor
    and build sites using the progressive enhancement methodology, 
    there are plenty of instances where I am arguably doing things in a graceful degradation manner. 
    '''
    content2 = '''
    W
    #wrapper div / Setting a context for proportional elements, The incredibly
    versatile max-width property
    W3C
    about / CSS3 enables responsive designs and more
    / How to write HTML5 pages
    W3C documentation
    on multiple background elements, URL / Background shorthand
    W3C HTML5 validator
    URL / Saving time and code with HTML5
    WAI-ARIA
    used, for adding accessibility to site / Adding accessibility to your site
    with WAI-ARIA
    WD / You can use media queries today
    Webkit (-webkit- ) / Vendor prefixes and how to use them
    Web Open Font Format (WOFF) / The @font-face CSS rule
    Webshims Lib
    URL for downloading / How to polyfill non-supporting browsers
    web typography
    about / Custom web typography
    week input type, HTML5 / week
    width / What can media queries test for?
    '''

    content3 = """aksdfoaheiadflasjf ksdlf"""
    
    print(fluency_score(content), fluency_score(content2), fluency_score(content3))
