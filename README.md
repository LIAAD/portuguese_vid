# Portuguese Variety Identifier  

A production-level solution for training and evaluating **delexicalized Portuguese Variety Identification (VID) models** following the AAAI 2025 paper: **Enhancing Portuguese Variety Identification with Cross-Domain Approaches**. In addition to the core training and evaluation pipelines, this repository also includes a simple [Streamlit](https://streamlit.io/) demo and a [FastAPI](https://fastapi.tiangolo.com/) endpoint.  

---

## Warning: Current Status  

**Important Notice:**  

- **Incomplete Repository:** This repository is incomplete. It represents an industry-level refactor of a scientific research project that was submitted to AAAI.  
- **Branch Archive:** The branch **"AAAI"** is an archival version and should not be extended.  

---

## Features  

- **Production-Ready Code:** Implements state-of-the-art delexicalized VID solutions.  
- **Research Integration:** Based on methods described in the AAAI 2025 paper.  
- **Modular Design:** Training and evaluation routines are packaged as a Python module.  
- **Interactive Demo & API:** Run a Streamlit demo and a FastAPI endpoint for quick model interaction.  
- **Docker Support:** Easily spin up the demo and API using Docker Compose.  
- **PyPI Package:** The `pt_vid` package is available on PyPI, ensuring smooth installation and integration via GitHub Actions pipelines.  
- **HuggingFace Compatibility:** Our best model is fully compatible with HuggingFace and runs off-the-shelf.  

---

## Quickstart: Using the HuggingFace Pipeline  

You can quickly run our best model using HuggingFace's pipeline API:  

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="liaad/PtVId")

result = pipe("Olá tudo bem? Este trabalho é só um ponto de partida")

print(result)
```  

This command instantiates the model and performs text classification immediately.  

---

## Getting Started  

### 1. Set Up a Virtual Environment  

We recommend using Conda for an isolated Python environment. **The recommended Python version is 3.10:**  

```sh
conda create --name .conda python=3.10
conda activate .conda
```  

### 2. Install Dependencies  

You can install the package directly from the source in editable mode:  

```sh
pip install -e .
```  

Alternatively, install the production-ready package from PyPI:  

```sh
pip install -U pt-vid
```  

---

## Training and Evaluation  

Examples of training and evaluation routines are provided in the `/exec` directory:  

- **Training:**  
  An example training script is available in [`/exec/Train.py`](exec/Train.py). This script demonstrates how to execute the training pipeline.  

- **Evaluation & Result Plotting:**  
  An example evaluation and result plotting script is available in [`/exec/Test.py`](exec/Test.py). This script shows how to evaluate the trained models and visualize the results.  

---

## Running the Demo and API  

A simple Streamlit demo and a FastAPI endpoint are included for quick testing and integration.  

### Using Docker Compose  

The recommended way to run the demo and API endpoints is via Docker Compose. Ensure Docker is installed, then run:  

```sh
docker-compose -f dev.docker-compose.yml up
```  

This command will launch both the Streamlit app and the FastAPI service in development mode.  

---

## Repository Structure  

- **`/exec`:** Contains scripts to execute training and evaluation routines.  
- **`/your_package`:** The main Python package implementing the VID solutions.  
- **`dev.docker-compose.yml`:** Docker Compose file for running the demo and API endpoints.  
- **Other directories/files:** Additional resources, utilities, and configurations.  

---

## Contribution and Future Work  

The goal of introducing industry/production-level code in this experiment was to establish the major guidelines for extending our work. Our aim is to deliver this framework to research teams around the globe, who will adapt the code to their respective languages and needs.  

Beyond making the code open-source, the authors also **intend to continue developing this package with low priority**, improving and extending it over time.  

### Future Increments (Without Time Estimate):  
- **Migration to Apache Airflow:** Move the scripts under the `/exec` directory to the Apache Airflow ecosystem to provide a more human-friendly track of the processes abstracted in these scripts.  
- **Completion of Missing Parts:** Implement the features covered in the AAAI submission but not yet integrated into this repository, including:  
  - Extending the dataset generators.  
  - Implementing the transformer-based training pipeline.  

It is expected that any future contributions adhere to the foundations established post-AAAI, which are maintained in the **main** branch. These contributions will help complete the missing components in this repository and further advance the project.  

---

## Citation  

If you use this model in your work, please cite the following paper:
```
@article{Sousa2025,
   author = {Hugo Sousa and Rúben Almeida and Purificação Silvano and Inês Cantante and Ricardo Campos and Alipio Jorge},
   doi = {10.1609/aaai.v39i24.34705},
   issn = {2374-3468},
   issue = {24},
   journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
   month = {4},
   pages = {25192-25200},
   title = {Enhancing Portuguese Variety Identification with Cross-Domain Approaches},
   volume = {39},
   year = {2025}
}
```

---

## License  

This project is licensed under the **MIT License**. Check the [LICENSE](LICENSE) file for further licensing information.  

---

## Contact  

For any questions or issues, please open an issue on GitHub or contact:  

- **Rúben Almeida** – [ruben.f.almeida@inesctec.pt](mailto:ruben.f.almeida@inesctec.pt)  
- **Hugo Sousa** – [hugo.o.sousa@inesctec.pt](mailto:hugo.o.sousa@inesctec.pt)  
