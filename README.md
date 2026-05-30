# AP M2 - Grupo5 | ERCP Image Classification

Classificação automática de imagens de ERCP (Colangiopancreatografia Retrógrada Endoscópica) em 4 classes.

**Classes:** `Lithiasis` · `Stricture` · `Normal` · `Biliary Leaks`  
**Melhor resultado:** EfficientNetV2-L com CLAHE — Teste Macro-F1 = **0.6918**

---

## Estrutura do repositório

```
AP-M2-G5/
├── filter_benchmark_efficientnet_b7 copy.ipynb           melhor modelo (CLAHE + geometry-aware)
├── filter_benchmark_efficientnet_b7 copy 3.ipynb        EfficientNetV2-L sem CLAHE
├── filter_benchmark_efficientnet_v2l_scratch_fixed(2).ipynb
├── filter_benchmark_efficientnet_v2l_scratch_fixed(2) copy.ipynb
├── filter_benchmark_efficientnet_v2l_scratch_fixed(2) copy 2.ipynb
├── filter_benchmark_efficientnet_v2l_folder_based.ipynb
├── efficientnet_v2_l_improved.ipynb                     EfficientNetV2-L com CLAHE (variante)
├── efficientnetv2s_manifest_splits_sem_focal_loss_corrigido.ipynb
├── efficientnetv2s_no_leakage_stratified_group_focal_complete.ipynb
├── efficientnetv2s_no_leakage_stratified_group_sem_focal_loss.ipynb
├── efficientnetv2s_no_leakage_threshold_tuning_complete.ipynb
├── densenet121_4class_sampler_finetune_completo.ipynb
├── mobilenet_v3_large_miqr_cc_notebook.ipynb
├── mobilenetv3_large_manifest_splits_no_focal.ipynb
├── regnety400mf_4class_no_sampler_corrigido_config.ipynb
├── regnety8gf_4class_ercp_pipeline.ipynb
├── AP_M2_EfficientNetB0_MONAI.ipynb
├── final_4class_ercp_pipeline.ipynb
├── filter_benchmark_clahe.ipynb                         benchmark de filtros de pré-processamento
├── EFICIENTNET.ipynb                               
│
├── requirements.txt
└── README.md
```

---

## Dataset

O dataset contém as seguintes colunas relevantes:

| Coluna | Descrição |
|---|---|
| `raw_image_path` | Caminho para a imagem original |
| `processed_image_path` | Caminho para a imagem pré-processada |
| `patient_id` | Identificador do paciente (usado para evitar data leakage nos splits) |
| `Label` | Label original (inclui `Unlabelled`, `Benign Stricture`, etc.) |
| `Keep` | Flag booleana — apenas imagens com `Keep=True` são usadas |
| `image_type` | Tipo de imagem (`S` = estático, `V` = vídeo) |

 A organização das labels para as 4 classes finais é:

```python
FUSED_LABEL_MAP = {
    'Benign Stricture':    'Stricture',
    'Malignant Stricture': 'Stricture',
    'Biliary Leaks':       'Biliary Leaks',
    'Lithiasis':           'Lithiasis',
    'Normal':              'Normal',
}
```

## Reprodução dos resultados

Os notebooks são independentes entre si. Cada um define os seus próprios caminhos, hiperparâmetros e pipeline de pré-processamento. Basta abrir o notebook pretendido, ajustar os caminhos de dados e correr todas as células em sequência.

## Resultados por modelo

Todos os valores são **Teste Macro-F1** no conjunto de teste.

| Modelo | Teste Macro-F1 | Notebook |
|---|---|---|
| EfficientNetV2-L com CLAHE (geometry-aware) | **0.6918** | `filter_benchmark_efficientnet_b7 copy.ipynb` |
| EfficientNetV2-L com CLAHE + TTA | 0.6511 | `efficientnet_v2_l_improved.ipynb` |
| EfficientNetV2-L com CLAHE sem TTA | 0.6485 | `efficientnet_v2_l_improved.ipynb` |
| EfficientNetV2-L sem CLAHE (geometry-aware, from scratch) | 0.6344 | `filter_benchmark_efficientnet_v2l_scratch_fixed(2).ipynb` |
| EfficientNetV2-L sem CLAHE (geometry-aware, pretrained) | 0.6344 | `filter_benchmark_efficientnet_v2l_scratch_fixed(2) copy.ipynb` |
| EfficientNetV2-L sem CLAHE (folder-based) | 0.5967 | `filter_benchmark_efficientnet_v2l_scratch_fixed(2) copy 2.ipynb` |
| MobileNetV3-Large (manifest splits) | 0.4675 | `mobilenetv3_large_manifest_splits_no_focal.ipynb` |
| EfficientNetV2-L sem CLAHE (geometry-aware) | 0.4443 | `filter_benchmark_efficientnet_b7 copy 3.ipynb` |
| RegNetY-8GF | 0.4403 | `regnety8gf_4class_ercp_pipeline.ipynb` |
| Final ERCP pipeline | 0.4298 | `final_4class_ercp_pipeline.ipynb` |
| EfficientNetV2-S com TTA (manifest splits) | 0.4206 | `efficientnetv2s_manifest_splits_sem_focal_loss_corrigido.ipynb` |
| EfficientNetV2-S com/sem Focal Loss (no leakage) | 0.4187 | `efficientnetv2s_no_leakage_stratified_group_*.ipynb` |
| MobileNetV3-Large (MIQR-CC) | 0.4013 | `mobilenet_v3_large_miqr_cc_notebook.ipynb` |
| EfficientNetV2-S sem Focal Loss (manifest splits) | 0.3625 | `efficientnetv2s_manifest_splits_sem_focal_loss_corrigido.ipynb` |
| DenseNet121 (com sampler, finetune) | 0.3372 | `densenet121_4class_sampler_finetune_completo.ipynb` |
| RegNetY-400MF | 0.3178 | `regnety400mf_4class_no_sampler_corrigido_config.ipynb` |
| EfficientNetB0 (MONAI) | 0.3136 | `AP_M2_EfficientNetB0_MONAI.ipynb` |
| EfficientNetV2-S threshold tuning + TTA | 0.2562 | `efficientnetv2s_no_leakage_threshold_tuning_complete.ipynb` |



## Principais técnicas utilizadas

**Pré-processamento**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) no espaço de cor LAB, aplicado ao canal L
- Geometry-aware preprocessing: `pad-to-square` antes do resize para preservar proporções originais das imagens ERCP
- Normalização ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

**Augmentação**
- `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` durante treino
- TTA (Test-Time Augmentation) em alguns notebooks

**Gestão de desequilíbrio de classes**
- `WeightedRandomSampler` no DataLoader
- Pesos de classe na loss (`CrossEntropyLoss` ponderada)
- `FocalLoss` como alternativa

**Arquiteturas testadas**
- EfficientNetV2-L, EfficientNetV2-S, EfficientNetB0, EfficientNetB7
- DenseNet121, DenseNet201
- MobileNetV3-Large
- RegNetY-400MF, RegNetY-8GF
- SEResNet50

---

