# 🧬 lncrnaPI - LncRNA–Protein Interaction Prediction

lncrnaPI is a command-line tool for predicting **lncRNA–Protein interactions** using **pre-trained language models (DNABERT-2 and ESM-2)** for sequence embedding and a **CatBoost classifier** for interaction probability estimation.

---

## 🚀 Overview

This standalone script enables large-scale prediction of interactions between **lncRNA** and **protein sequences**.  
It leverages state-of-the-art transformer models to extract biologically meaningful embeddings and a **CatBoost** model to compute lncRNA-Protein interaction probabilities.

---

## 📦 Features

- Supports **FASTA** input for lncRNA and protein sequences.  
- Generates embeddings using:
  - 🧬 **DNABERT-2** (`zhihan1996/DNABERT-2-117M`) for lncRNAs  
  - 🧫 **ESM-2** (`facebook/esm2_t30_150M_UR50D`) for proteins  
- Predicts interaction probabilities using a **CatBoost classifier**.  
- Supports GPU acceleration (**CUDA** / **MPS**) for faster inference.  
- Outputs results in **CSV** format.

---

## 🧰 Dependencies

Install the following dependencies before running the script:

```bash
pip install torch transformers catboost joblib tqdm numpy pandas
```
---

## ⚙️ Usage

Run the script directly from the command line:

```bash
python lncrnapi_cli.py  --lncrna_fasta /path/to/lncrnas.fasta   --protein_fasta /path/to/proteins.fasta   --model_path /path/to/saved_model.joblib   --output_file /path/to/results.csv
```

### **Arguments**

| Argument | Description | Required |
|-----------|--------------|-----------|
| `--lncrna_fasta` | Path to the FASTA file containing lncRNA sequences. | ✅ |
| `--protein_fasta` | Path to the FASTA file containing protein sequences. | ✅ |
| `--model_path` | Path to the pre-trained CatBoost model file (`.cbm`, `.joblib`, or `.pkl`). | ✅ |
| `--output_file` | Path to save the CSV file with predicted probabilities. | ✅ |

---

## 🧠 How It Works

1. **Model Loading**  
   The tool loads the DNABERT-2 and ESM-2 models from Hugging Face.

2. **FASTA Parsing**  
   Extracts sequence IDs and corresponding sequences from input FASTA files.

3. **Embedding Generation**  
   Computes mean pooled embeddings for each sequence using transformer hidden states.

4. **Prediction**  
   Concatenates embeddings (lncRNA + protein) and predicts the interaction probability using the CatBoost model.

5. **Output**  
   Generates a `.csv` file containing:
   - `LncRNA_ID`
   - `Protein_ID`
   - `Interaction_Probability`

---

## 📊 Example Output

| LncRNA_ID | Protein_ID | Interaction_Probability |
|------------|-------------|--------------------------|
| lnc001 | P12345 | 0.9421 |
| lnc002 | Q8N6T7 | 0.3175 |

---

## ⚡ Hardware Acceleration

The script automatically detects and uses available hardware:

- ✅ **CUDA GPU** (NVIDIA)
- ✅ **MPS** (Apple Silicon)
- ⚠️ **CPU** (fallback)

---

## 🧩 Model Formats Supported

| Format | Description |
|---------|-------------|
| `.cbm` | Native CatBoost model format |
| `.joblib` | Joblib-serialized model |
| `.pkl` | Pickle-based serialized model |

---

## 🛠 Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|-----------|
| `Model file not found` | Wrong `--model_path` | Check the file path |
| `No sequences found in FASTA` | Invalid FASTA format | Ensure `>` headers are present |
| `safetensors` error | Missing library | Install with `pip install safetensors` |
| Slow performance | CPU usage | Use GPU-enabled environment |

---

## 📁 Output Example

```bash
$ head results.csv
LncRNA_ID,Protein_ID,Interaction_Probability
lnc001,P12345,0.9421
lnc002,Q8N6T7,0.3175
lnc003,O76074,0.7814
```

---

## 📜 Citation

If you use this tool in your research, please cite:

> **XXX et. al.**  
> *A Deep Learning Framework for lncRNA–Protein Interaction Prediction Using Transformer-Based Sequence Embeddings* (2025)

---

## 🧩 Repository Structure

```
├── predict_interaction.py       # Main CLI script
├── README.md                    # Documentation
└── example/
    ├── lncrnas.fasta
    ├── proteins.fasta
    └── saved_model.joblib
```

---
