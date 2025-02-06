from pt_vid.data.generators.GenerateLaw import GenerateLaw
from pt_vid.data.generators.GenerateWeb import GenerateWeb
from pt_vid.data.generators.GenerateNews import GenerateNews
from pt_vid.data.generators.GeneratePolitics import GeneratePolitics
from pt_vid.data.generators.GenerateLiterature import GenerateLiterature
from pt_vid.data.generators.GenerateSocialMedia import GenerateSocialMedia


law_dataset = GenerateLaw().generate()
web_dataset = GenerateWeb().generate()
news_dataset = GenerateNews().generate()
politics_dataset = GeneratePolitics().generate()
social_media_dataset = GenerateSocialMedia().generate()
literature_dataset = GenerateLiterature().generate()

