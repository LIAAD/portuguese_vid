from tqdm import tqdm
from pt_vid.data.Cleaner import Cleaner
from pt_vid.data.Sampler import Sampler
from pt_vid.data.Splitter import Splitter
from pt_vid.entity.CorporaStats import CorporaStats
from pt_vid.data.generators.GenerateWeb import GenerateWeb
from pt_vid.data.generators.GenerateNews import GenerateNews
from pt_vid.data.splitters.DefaultSplitterStrategy import DefaultSplitterStrategy

domains = {}


for domain in tqdm([
        #GenerateLaw, 
        GenerateWeb, GenerateNews, 
        #GeneratePolitics, 
        #GenerateLiterature, 
        #GenerateSocialMedia
    ]):
    domain_instance = domain().generate()
    domains[domain_instance.config_name] = domain_instance


corpora_stats = CorporaStats(
    dataset_stats=[domain.dataset_stats for domain in domains.values()]
)

print(corpora_stats.model_dump())

# Clean the dataset (create additional column)
web_dataset = Cleaner.run(domains['web'].dataset, domain='web')

# Split the dataset
splitter = Splitter(strategy=DefaultSplitterStrategy)

dataset_dict = splitter.run(web_dataset, domain='web')

# Sample the dataset
dataset_dict = Sampler.run(dataset_dict)

# Save based on multiple_configs
print(dataset_dict)