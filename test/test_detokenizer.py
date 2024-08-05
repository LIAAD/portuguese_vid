from ptvid.data.detokenizer import PortugueseDetokenizer


def test_detokenize():
    detokenizer = PortugueseDetokenizer()

    tokens = []
    assert detokenizer.detokenize(tokens) == ""

    tokens = ["Hello"]
    assert detokenizer.detokenize(tokens) == "Hello"

    tokens = ["Hello", "world", "!"]
    assert detokenizer.detokenize(tokens) == "Hello world!"

    tokens = ["I", "don't", "like", "it"]
    assert detokenizer.detokenize(tokens) == "I don't like it"

    tokens = ["(", "Hello", ")", "[", "world", "]"]
    assert detokenizer.detokenize(tokens) == "(Hello) [world]"

    tokens = ["Hello", ",", "world", "!"]
    assert detokenizer.detokenize(tokens) == "Hello, world!"

    tokens = ["``", "Hello", "world", "''"]
    assert detokenizer.detokenize(tokens) == '"Hello world"'

    tokens = ["I", "-me", "like", "you", "-te"]
    assert detokenizer.detokenize(tokens) == "I-me like you-te"

    tokens = ["Hello", "...", "world"]
    assert detokenizer.detokenize(tokens) == "Hello... world"

    tokens = ["Hello", ":", "world", ";"]
    assert detokenizer.detokenize(tokens) == "Hello: world;"

    tokens = "Tanto é assim , que os próprios créditos trabalhistas que dão origem às contribuições previdenciárias sobre eles incidentes estão sujeitos a atualização monetária e juros da mora , apesar da eventual existência de controvérsia .".split()
    assert detokenizer.detokenize(tokens) == "Tanto é assim, que os próprios créditos trabalhistas que dão origem às contribuições previdenciárias sobre eles incidentes estão sujeitos a atualização monetária e juros da mora, apesar da eventual existência de controvérsia."


def test_detokenize_quotes():
    detokenizer = PortugueseDetokenizer()
    tokens = """Coloquio " O seculo XX portugues: imagens , discursos e personalidades " , na Casa de Serralves""".split()
    assert detokenizer.detokenize(tokens) == """Coloquio "O seculo XX portugues: imagens, discursos e personalidades", na Casa de Serralves"""

    tokens = """ " O seculo XX portugues: imagens , discursos e personalidades " , na Casa de Serralves""".split()
    assert detokenizer.detokenize(tokens) == """"O seculo XX portugues: imagens, discursos e personalidades", na Casa de Serralves"""
    
    tokens = ' " O seculo XX portugues: imagens , discursos e personalidades " '.split()
    assert detokenizer.detokenize(tokens) == '"O seculo XX portugues: imagens, discursos e personalidades"'
    
    tokens = '" O seculo XX portugues: imagens , discursos e personalidades " '.split()
    assert detokenizer.detokenize(tokens) == '"O seculo XX portugues: imagens, discursos e personalidades"'
    
    tokens = '" O seculo XX portugues: imagens , discursos e personalidades "'.split()
    assert detokenizer.detokenize(tokens) == '"O seculo XX portugues: imagens, discursos e personalidades"'
    