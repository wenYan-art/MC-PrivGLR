# Federated Learning with Profile Mapping under Distribution Shifts and Drifts

This is the official implementation of the paper accepted at **ICLR 2026**:

> **Federated Learning with Profile Mapping under Distribution Shifts and Drifts**  
> Mohan Li, Dario Fenoglio, Martin Gjoreski, Marc Langheinrich

---


## 🌍 Overview

**FEROMA** is a lightweight Federated Learning (FL) framework that addresses **distribution shift across clients** and **distribution drift over time** by guiding aggregation through compact, privacy-preserving **distribution profiles**. Instead of relying on explicit clustering or client identities, FEROMA dynamically maps clients across training rounds based on profile similarity. Unlike prior FL approaches, FEROMA:
- requires **no prior knowledge** of the number of clusters or heterogeneity types,  
- handles **both shift and drift** during training and at test time, and  
- supports **one-shot test-time model assignment** for unseen and unlabeled clients, without retraining.  

Extensive experiments across six datasets (MNIST, FMNIST, CIFAR-10, CIFAR-100, CheXpert, Office-Home) show that FEROMA significantly improves robustness under heterogeneous and non-stationary data, while keeping computation and communication overhead comparable to FedAvg.

<p align="center">
  <img src="plots/feroma_overview.png" alt="FEROMA Overview" width="100%"/>
</p>

---

## 📦 Key Features
- **Distribution Profile Extraction:** Client-side extraction of compact, differentially-private profiles that summarize local data distributions and enable similarity-based collaboration.  
  Captures:
  - Feature distribution shifts: \(P(X)\) varies across clients.  
  - Label distribution shifts: \(P(Y)\) varies across clients.  
  - Concept shifts (same features, different labels): \(P(Y|X)\) varies.  
  - Concept shifts (same labels, different features): \(P(X|Y)\) varies.  
- **Profile-Guided Aggregation:** Similarity-based, weighted aggregation across rounds that automatically recovers clustered, personalized, or global FL behaviours.  
- **Test-Time Model Assignment:** One-shot assignment of unseen, unlabeled clients to the most suitable trained model using label-free profiles.  
- **Scalability & Efficiency:** Minimal computation and communication overhead, comparable to FedAvg, enabling scalable deployment in large and dynamic FL systems.  

<p align="center">
  <img src="plots/drift_shift.png" alt="Distribution shifts and drifts in Federated Learning" width="100%"/>
</p>

---

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dariofenoglio98/FEROMA.git
   cd FEROMA
   ```
2. (Optional) Clone ANDA only if you plan to use `public/generate_datasets.py` for MNIST/FMNIST/CIFAR experiments:
   ```bash
   git clone https://github.com/alfredoLimo/ANDA.git
   ```
   Keep the folder name as `ANDA` at repository root.
3. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. If running CheXpert generation (`public/chexpert_data_gen.py`), set your local CheXpert dataset path in that script (`path = ...`).

---

## ⚙️ Configuration
Main settings live in `public/config.py`:
- `strategy`: `fedavg` or `feroma` (must match folder names).
- `dataset_name`: `MNIST`, `FMNIST`, `CIFAR10`, `CIFAR100`, or `CheXpert`.
- `model_name`: `LeNet5` or `ResNet9`.
- Shift/drift setup: `non_iid_type` (`Px`, `Py`, `Px_y`, `Py_x`) and `drifting_type`.
- Training setup: `n_clients`, `n_rounds`, `local_epochs`, `lr`, batch sizes, etc.

---

## 🏃‍♂️ Running Experiments
Once configured, launch experiments with:
```bash
bash run.sh
```
The script will:
- generate/update datasets,
- enter the selected strategy folder (`fedavg/` or `feroma/`),
- start one server and `n_clients` clients.

Artifacts are saved under:
- `<strategy>/results/<seed>/<model>/<dataset>/<drifting_type>/`
- `<strategy>/histories/<seed>/<model>/<dataset>/<drifting_type>/`
- `<strategy>/images/<seed>/<model>/<dataset>/<drifting_type>/`
- `<strategy>/checkpoints/<seed>/<model>/<dataset>/<drifting_type>/`

---

## License
This project is licensed under the MIT License – see the LICENSE.md file for details.

---

## Citation
If you use this code, please cite our paper:
```bibtex
@inproceedings{li2025feroma,
  title     = {Federated Learning with Profile Mapping under Distribution Shifts and Drifts},
  author    = {Li, Mohan and Fenoglio, Dario and Gjoreski, Martin and Langheinrich, Marc},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
