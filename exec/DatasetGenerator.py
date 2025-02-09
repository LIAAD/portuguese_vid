from tqdm import tqdm
from pt_vid.data.Cleaner import Cleaner
from pt_vid.entity.CorporaStats import CorporaStats
from pt_vid.data.generators.GenerateWeb import GenerateWeb
from pt_vid.data.generators.GenerateNews import GenerateNews

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
Cleaner.run(domains['web'].dataset)


# Split the dataset

# Sample the dataset

# Save based on multiple_configs