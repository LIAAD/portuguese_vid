from datasets import load_dataset, Dataset, Features, Value, ClassLabel, concatenate_datasets
import pandas as pd
from tqdm import tqdm
import re
import dotenv
import os


oscar = load_dataset('oscar-corpus/OSCAR-2201', 'pt',
                     split='train', streaming=True)

final_dataset = []

dotenv.load_dotenv(dotenv.find_dotenv())

def filter_websites(url):
    
    list_valid_prefixes = None

    if '.pt' in url:
        variant = 'pt-PT'

        list_valid_prefixes = [
            '.sapo.pt',
            '.unl.pt',
            '.iscte-iul.pt',
            '.ulusofona.pt',
            '.ipcb.pt',
            '.ipg.pt',
            '.ipportalegre.pt',
            '.ipsantarem.pt',
            '.ipt.pt',
            '.ipleiria.pt',
            '.ipvc.pt',
            '.iscte.pt',
            '.iseg.ulisboa.pt',
            '.uc.pt',
            '.up.pt',
            '.utad.pt',
            '.uma.pt',
            '.ubi.pt',
            '.uevora.pt',
            '.unl.pt',
            '.pgdlisboa.pt',
            '.jn.pt',
            '.publico.pt',
            '.dn.pt',
            '.rtp.pt',
            '.sic.pt',
            '.cmjornal.pt',
            '.abola.pt',
            '.ojogo.pt',
            '.record.pt',
            '.maisfutebol.iol.pt',
            '.noticiasdecoimbra.pt',
            '.noticiasdeaveiro.pt',
            '.noticiasdeleiria.pt',
            '.observador.pt',
            '.jornaldenegocios.pt',
            '.dinheirovivo.pt',
            '.eco.pt',
            '.expresso.pt',
            '.visao.pt',
            '.sabado.pt',
            '.ionline.pt',
            '.jornaleconomico.pt',
            '.negocios.pt',
            '.jornaldenegocios.pt',
            '.dinheirovivo.pt',
            '.iol.pt',
            '.ps.pt',
            '.psd.pt',
            '.cds.pt',
            '.pcp.pt',
            '.timeout.pt',
            '.wook.pt',
            '.bertrand.pt',
            '.fnac.pt',
            '.ucp.pt',
            '.idealista.pt',
            '.remax.pt',
            '.era.pt',
            '.century21.pt',
            '.kwportugal.pt',
            '.cienciaviva.pt',
            '.ulisboa.pt',
            '.sicnoticias.pt',
            '.tsf.pt',
            '.eurogamer.pt',
            '.pplware.sapo.pt',
            '.tek.sapo.pt',
            '.shifter.pt',
            '.cmjornal.pt',
            '.aeiou.pt',
            '.pan.com.pt',
            '.portoeditora.pt',
            '.olx.pt',
            '.cm-amadora.pt',
            '.cm-murtosa.pt',
            '.cm-almada.pt'
            '.cm-amadora.pt'
            '.cm-barreiro.pt'
            '.cm-cascais.pt'
            '.cm-coimbra.pt'
            '.cm-evora.pt'
            '.cm-faro.pt'
            '.cm-gondomar.pt'
            '.cm-guimaraes.pt'
            '.cm-leiria.pt'
            '.cm-lisboa.pt'
            '.cm-loures.pt'
            '.cm-maia.pt'
            '.cm-matosinhos.pt'
            '.cm-oeiras.pt'
            '.cm-palmela.pt'
            '.cm-paredes.pt'
            '.cm-ponta-delgada.pt'
            '.cm-porto.pt'
            '.cm-portimao.pt'
            '.cm-santarem.pt'
            '.cm-setubal.pt'
            '.cm-sintra.pt'
            '.cm-vilanovadegaia.pt'
            '.cm-vilarealdesantoantonio.pt'
            '.cm-viseu.pt',
            '.gov.pt',
            '.cm-aveiro.pt',
            '.nit.pt',
            '.flash.pt',
            '.cgd.pt',
            '.sabado.pt',

        ]
    elif '.br' in url:
        variant = 'pt-BR'
        list_valid_prefixes = [
            '.uol.com.br',
            '.estadao.com.br',
            '.terra.com.br',
            '.ig.com.br',
            '.veja.abril.com.br',
            '.mercadolivre.com.br',
            '.tecmundo.com.br',
            '.olhardigital.com.br',
            '.olx.com.br',
            '.fazenda.gov.br',
            '.vivo.com.br',
            '.gov.br',
            '.amazon.com.br',
            '.usp.br',
            '.carrefour.com.br',
            '.unicamp.br',
            '.ufpr.br',
            '.universal.org',
            '.bradesco.com.br',
            '.itau.com.br',
            '.caixa.gov.br',
            '.santander.com.br',
            '.bancobrasil.com.br',
            '.bb.com.br',
            '.hsbc.com.br',
            '.espn.com.br',
        ]
    else:
        com_prefixes = [
            '.noticiasaominuto.com',
            '.noticiasdevilareal.com',
            '.comunidadeculturaearte.com',
            '.leyaonline.com',
            '.imovirtual.com',
            '.esquerda.net',
            '.leya.com',
            '.leyaeducacao.com',
            '.leyaonline.com',
            '.globo.com',
            '.r7.com',
        ]
        variants = [
            'pt-PT',
            'pt-PT',
            'pt-PT',
            'pt-PT',
            'pt-PT',
            'pt-PT',
            'pt-PT',
            'pt-PT',
            'pt-PT',
            'pt-BR',
            'pt-BR',
        ]

    if list_valid_prefixes is not None:

        for prefix in list_valid_prefixes:
            if prefix in url:
                return variant

    else:
        for prefix in com_prefixes:
            if prefix in url:
                return variants[com_prefixes.index(prefix)]
            
    return None


def beautify_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')

    return text


websites = {
    'pt-PT': [],
    'pt-BR': []
}

for line in tqdm(oscar):
    variant = None

    if len(websites['pt-BR']) >= 10000 and len(websites['pt-PT']) >= 10000:
        break

    uri = line['meta']['warc_headers']['warc-target-uri']

    if (variant := filter_websites(uri)) is not None:
        websites[variant].append(beautify_text(line['text']))

    if variant is not None and len(websites[variant]) % 1000 == 0 and len(websites[variant]) > 0:
        print(f"Length: {len(websites[variant])} Variant: {variant}")

for variant in ['pt-PT', 'pt-BR']:
    df = pd.DataFrame(websites[variant], columns=['text'])

    df['label'] = variant

    final_dataset.append(Dataset.from_pandas(df, split='train', features=Features({
        "text": Value("string"),
        "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
    })))

final_dataset = concatenate_datasets(
    [final_dataset[0], final_dataset[1]], split='train')

final_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).push_to_hub(
    'portuguese-oscar-li', token=os.getenv("HF_TOKEN"), private=False
)
