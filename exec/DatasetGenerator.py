from pt_vid.data.generators.GenerateLaw import GenerateLaw
from pt_vid.data.generators.GeneratePolitics import GeneratePolitics
from pt_vid.data.generators.GenerateNews import GenerateNews
from pt_vid.data.generators.GenerateSocialMedia import GenerateSocialMedia


law_dataset = GenerateLaw().generate()
politics_dataset = GeneratePolitics().generate()
news_dataset = GenerateNews().generate()
social_media_dataset = GenerateSocialMedia().generate()