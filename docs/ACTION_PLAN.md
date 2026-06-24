# Plan de Acción — Tesis Doctoral Diana Ayala Roldán

## Estado Actual (Diagnóstico)

### Lo que YA está hecho
- Escritura completa de los 6 capítulos (estructura, narrativa, ecuaciones, tablas comparativas)
- Abstract en inglés y español (con placeholders numéricos)
- References.bib con ~90 entradas (algunas con autores genéricos que corregir)
- Diseño metodológico completo de los 6 módulos
- **✅ TODO el código implementado y testeado** (CNN-Transformer, 3 decoders 2D, VolumetricWoundEncoder3D, PolarDecoder3DLayered, trajectory planner, IK solver)
- **✅ Notebooks Kaggle listos** para generar resultados (01_ablation + 02_volumetric)
- **✅ Tesis actualizada** con sección de reconstrucción volumétrica CT-style
- **✅ Structured logging** en todos los módulos

### Lo que FALTA (todo lo crítico)
- **104 valores numéricos en rojo** en Results que necesitan datos reales
- **34 valores en rojo** en Discussion que dependen de los de Results
- **11 valores en rojo** en Conclusions
- **23 figuras PLACEHOLDER** que necesitan imágenes reales de experimentos
- **Correr notebooks en Kaggle GPU** (2-3 horas cada uno → genera todos los datos)
- **5 notas "NOTE TO FUTURE ME"** pendientes

---

## Fases de Trabajo

---

### FASE 0: Setup del Entorno — ✅ COMPLETADA

**Estado:** COMPLETADA. Repo en GitHub, entorno funcional, tests pasan.

- [x] **0.1** Crear repositorio Git → `github.com/dianisay/diana-bioprinting-pipeline`
- [x] **0.2** Configurar entorno Python (PyTorch, torchvision, matplotlib, scipy, PuLP, Open3D, OpenCV)
- [ ] **0.3** Configurar CoppeliaSim + ZeroMQ remote API (Python) — pendiente
- [ ] **0.4** Descargar y preparar dataset FUSeg (1,210 imágenes, usar las 934 filtradas) — pendiente (usando synthetic por ahora)
- [ ] **0.5** Verificar que el modelo URDF del UR5 + gantry funciona en CoppeliaSim — pendiente
- [x] **0.6** Estructura de carpetas definida (models/, modules/, training/, data/, tests/, notebooks/, utils/)

---

### FASE 1: PoC Baseline — U-Net + G-Code (COMPLETADA)

**Estado:** COMPLETADA durante 2025.

Incluye: U-Net entrenada en FUSeg, pipeline G-Code funcional, simulación en CoppeliaSim con IK y PID. Los resultados del PoC sirven como baseline de comparación para el sistema propuesto (Tabla 4.11 del documento).

**Pendiente de esta fase:** Verificar que los datos del PoC ya estén reflejados en los placeholders de la Sección 4.1, o extraer las métricas finales si aún no se han sustituido.

---

### FASE 2: Módulo 1 — CNN-Transformer con Polar Decoder — ✅ CÓDIGO COMPLETO

**Estado:** Todo implementado y testeado. Falta correr `01_ablation_study_kaggle.ipynb` en GPU.

**Prioridad:** CRÍTICA — es LA contribución principal (2D ablation component).

#### 2.1 Preparación de Datos ✅
- [x] Implementar generador de wounds sintéticas (2,000 imágenes): `data/synthetic_generator.py`
- [x] Conversión polar↔cartesiano: `data/polar_conversion.py` (test verified)
- [x] Split train/val/test (70/15/15): `data/dataset.py`

#### 2.2 Encoder (ResNet-50 + Transformer) ✅
- [x] ResNet-50 backbone (pretrained ImageNet) → `models/encoder.py`
- [x] 1x1 Conv projection → 256-dim
- [x] Learned 2D positional encoding
- [x] Transformer encoder: 6 layers, 8 heads, d=256
- [x] Forward pass verified: `tests/test_models.py`

#### 2.3 Decoder v3 — Polar (propuesto) ✅
- [x] Centroid head + Radii head: `models/polar_decoder.py`
- [x] Loss combinada: L_centroid + L_radii + L_points

#### 2.4 Decoder v1 — DETR-style ✅
- [x] N learned query vectors + cross-attention + Hungarian loss: `models/detr_decoder.py`

#### 2.5 Decoder v2 — Autoregressive ✅
- [x] Teacher forcing + autoregressive decoder: `models/autoregressive_decoder.py`

#### 2.6 Entrenamiento y Ablation
- [x] Training loop con early stopping: `training/train.py`
- [x] Evaluation pipeline: `training/evaluate.py`
- [x] Ablation runner: `training/ablation.py`
- [x] Kaggle notebook listo: `notebooks/01_ablation_study_kaggle.ipynb`
- [ ] **⏳ PENDIENTE: Correr notebook 01 en Kaggle GPU (~2-3 horas)**
- [ ] **DATOS A EXTRAER (reemplazan los rojos del ablation en Ch.4):**
  - Para cada decoder: Chamfer dist, Hausdorff dist, IoU, closure error, ordering %
  - Training/val loss curves
  - Epoch de convergencia
  - Grid cualitativo de predicciones

---

### FASE 3: Módulo 2 — Reconstrucción Volumétrica CT-Style — ✅ CÓDIGO COMPLETO

**Estado:** Implementado como learned approach (no MVS clásico). Falta correr `02_volumetric_ablation_kaggle.ipynb` en GPU.

**Prioridad:** ALTA — es la PRINCIPAL contribución novel de la tesis.

**Decisión de diseño:** En lugar de MVS clásico con CoppeliaSim, se implementó un encoder volumétrico CT-style que aprende directamente la reconstrucción 3D end-to-end. Esto es más novedoso academicamente y genera output estructurado (boundary + depth + layer-fill) listo para bioprinting.

- [x] VolumetricWoundEncoder3D: 8× ResNet-18 + volumetric fusion + 3D Transformer → `models/volumetric_encoder.py`
- [x] PolarDecoder3DLayered: predice boundary + depth + layer-fill → `models/volumetric_decoder.py`
- [x] MultiViewWoundDataset: genera datos sintéticos multi-view → `data/multiview_dataset.py`
- [x] VolumetricWoundLoss: boundary MSE + depth MAE + layer BCE → `models/volumetric_decoder.py`
- [x] Bug fix: channel mismatch en ResNet18Backbone (corregido y testeado)
- [x] Kaggle notebook listo: `notebooks/02_volumetric_ablation_kaggle.ipynb`
- [ ] **⏳ PENDIENTE: Correr notebook 02 en Kaggle GPU (~1-2 horas)**
- [ ] **DATOS A EXTRAER:**
  - Boundary Chamfer distance (mm)
  - Depth MAE (mm)
  - Layer-fill accuracy / BCE loss
  - Honeycomb feasibility (%)
  - Inference time (ms)
  - Comparison vs traditional MVS (speed + structured output)

---

### FASE 4: Módulo 3 — Generación de Trayectoria 3D — ✅ CÓDIGO COMPLETO

**Estado:** Todo implementado y testeado (`tests/test_trajectory_pipeline.py` pasa).

**Prioridad:** ALTA — segunda contribución principal.

#### 4.1 Conformal Mapping ✅
- [x] Kasa circle fit para superficies cilíndricas: `modules/conformal_mapping.py`
- [x] Conformal map: superficie 3D → rectángulo (u,v)
- [x] UV → XYZ inverse mapping + normal computation

#### 4.2 Honeycomb Lattice ✅
- [x] Grid hexagonal: `modules/honeycomb.py`
- [x] Centroides de celdas + perimeter generation

#### 4.3 TSP/MILP Optimization ✅
- [x] MILP con MTZ constraints + PuLP: `modules/tsp_solver.py`
- [x] Dummy depot node (open-path → closed-tour)
- [x] Cost matrix: Euclidean + rise penalty
- [x] Test: ~7.5% travel reduction on test case

#### 4.4 Per-Cell Toolpath + 3D Mapping ✅
- [x] Full trajectory planner: `modules/trajectory_planner.py`
- [x] Outline + fill + deposit trajectories
- [x] UV → XYZ + nozzle orientations + workspace transform
- [x] Test: 23,520 trajectory points generated correctly

- [ ] **DATOS A EXTRAER (cuando se corra end-to-end con datos reales):**
  - Wound coverage %
  - Travel-to-deposition ratio
  - TSP travel reduction % (ya verificado ~7.5% en test)
  - Total waypoints per wound

---

### FASE 5: Módulo 4 — Motion Planning y Control — ✅ CÓDIGO COMPLETO

**Estado:** IK + APF + Super-Twisting implementados y testeados. CoppeliaSim integration pendiente.

**Prioridad:** ALTA — cierra el loop de ejecución.

- [x] Modelo cinemático 8-DOF (2P + 6R UR5): `modules/robot_model.py`
- [x] FK + Jacobian + Manipulability
- [x] IK numérico multi-seed con DLS: `modules/inverse_kinematics.py`
- [x] APF (Artificial Potential Fields) para limit avoidance
- [x] Super-Twisting sliding-mode control refinement
- [x] Test: 5/5 IK solutions found with <10mm error
- [ ] PID velocity controller (pendiente para CoppeliaSim)
- [ ] Ejecutar trayectorias en CoppeliaSim (Python ZeroMQ API)
- [ ] **DATOS A EXTRAER (cuando se integre con CoppeliaSim):**
  - IK success rate (% waypoints resueltos)
  - Manipulability: mean/min μ
  - Tracking error: mean/RMS/max position
  - Mean orientation error

---

### FASE 6: Módulo 5 — Ejecución Closed-Loop (1-2 semanas)

**Objetivo:** Monitoreo visual durante impresión y verificación post-deposición.

**Prioridad:** MEDIA-ALTA — completa el pipeline pero depende de M1-M4.

- [ ] Implementar captura periódica durante ejecución (cada ~5 celdas)
- [ ] Re-evaluar wound coverage con CNN-Transformer en cada captura
- [ ] Implementar post-deposition verification (imagen final → Module 1 → coverage restante)
- [ ] **DATOS A EXTRAER:**
  - Planned coverage vs measured coverage
  - Coverage gap
  - Secuencia visual de cobertura progresiva

---

### FASE 7: Validación End-to-End (1-2 semanas)

**Objetivo:** Correr el pipeline completo de principio a fin, sin intervención, en las 20 wounds.

**Prioridad:** ALTA — es la prueba final.

- [ ] Pipeline automatizado: imagen → M1 → M2 → M3 → M4 → M5 → métricas
- [ ] Correr en 20 wounds sintéticas
- [ ] **DATOS A EXTRAER (Tabla e2e_summary):**
  - Todos los valores módulo por módulo
  - End-to-end pipeline time
  - Post-deposition coverage
- [ ] Comparación directa PoC baseline vs Proposed (Table poc_comparison)

---

### FASE 8 (TENTATIVA): Validación con Phantoms Físicos

**Objetivo:** Tests IRL. Deseable pero NO bloqueante para la tesis.

**Prioridad:** BAJA — nice-to-have, no es vital.

- [ ] Imprimir 3-5 phantoms en FDM (PLA) con geometrías de wound variadas
- [ ] Ejecutar pipeline con cámara real
- [ ] Depositar con marker ink como proxy de bioink
- [ ] Medir Chamfer, coverage, tracking error en phantoms
- [ ] Si no se hace: ajustar texto de Ch.4-6 para reflejar que phantom validation es trabajo futuro

---

### FASE 9: Cierre del Documento (2-3 semanas, en paralelo con fases finales)

**Objetivo:** Reemplazar todos los placeholders con datos reales y generar figuras definitivas.

#### 9.1 Reemplazar valores numéricos
- [ ] Ch.4 Results: sustituir los ~104 `\textcolor{red}{...}` con valores experimentales reales
- [ ] Ch.5 Discussion: actualizar ~34 valores que referencian resultados
- [ ] Ch.6 Conclusions: actualizar ~11 valores
- [ ] Abstract (EN + ES): actualizar valores clave

#### 9.2 Generar figuras reales (23 PLACEHOLDERs)
- [ ] Fig. pipeline_overview — diagrama del pipeline completo
- [ ] Fig. hardware_arch — arquitectura hardware 8-DOF
- [ ] Fig. cnn_transformer_arch — diagrama de la arquitectura CNN-Transformer
- [ ] Fig. ablation_comparison — comparación cualitativa de 3 decoders
- [ ] Fig. unet_training — curvas de loss U-Net
- [ ] Fig. unet_validation — grid de segmentaciones cualitativas
- [ ] Fig. gcode_generation — pasos del pipeline G-code
- [ ] Fig. gcode_output — trayectoria G-code en frame del robot
- [ ] Fig. poc_sim_execution — screenshots CoppeliaSim del PoC
- [ ] Fig. m1_training — curvas de loss CNN-Transformer
- [ ] Fig. m1_qualitative — predicciones cualitativas polar decoder
- [ ] Fig. m2_reconstruction — reconstrucción 3D + heatmap error
- [ ] Fig. m2_multiview — diagrama multi-view acquisition
- [ ] Fig. m3_honeycomb — lattice en (u,v) y en 3D
- [ ] Fig. m3_toolpath — toolpath 3D completo
- [ ] Fig. honeycomb_lattice — lattice generation
- [ ] Fig. tsp_optimization — naive vs optimizado
- [ ] Fig. per_cell_toolpath — secuencia 5-phase
- [ ] Fig. m4_manipulability — perfil de manipulability
- [ ] Fig. m4_tracking — commanded vs achieved
- [ ] Fig. m5_monitoring — monitoreo progresivo
- [ ] Fig. feedback_loop — diagrama closed-loop
- [ ] Fig. phantom_results — (solo si se hace Fase 8)

#### 9.3 Limpieza final
- [ ] Corregir entradas .bib con autores genéricos ("Researcher Names", "Author Names Unknown", "Various Authors") — hay ~6 entradas así
- [ ] Resolver las 5 notas "NOTE TO FUTURE ME":
  - Antecedents: verificar papers concurrentes 2025-2028
  - Results M2: confirmar approach elegido
  - Results M5: actualizar con datos de monitoreo
  - Results phantoms: actualizar o mover a future work
  - Discussion: verificar papers concurrentes
  - Conclusions: listar publicaciones reales
- [ ] Verificar citation-needed en Discussion (clinical procedure time)
- [ ] Compilar LaTeX completo y verificar formato
- [ ] Revisar consistencia de todos los cross-references entre capítulos

---

## Cronograma Actualizado (Junio 2026)

| Fase | Estado | Lo que falta |
|------|--------|--------------|
| F0: Setup | ✅ COMPLETADA | — |
| F1: PoC Baseline | ✅ COMPLETADA (2025) | — |
| F2: CNN-Transformer 2D | ✅ CÓDIGO COMPLETO | Correr notebook 01 en Kaggle (~3h) |
| F3: Volumetric CT-style | ✅ CÓDIGO COMPLETO | Correr notebook 02 en Kaggle (~2h) |
| F4: Trayectoria 3D | ✅ CÓDIGO COMPLETO | Métricas con datos reales |
| F5: Motion Planning | ✅ CÓDIGO COMPLETO | CoppeliaSim integration |
| F6-F7: Execution + E2E | ❌ Pendiente | Depende de CoppeliaSim |
| F8: Phantoms | ❌ Pendiente (future work) | No bloqueante |
| F9: Cierre documento | En progreso | Valores reales + figuras |

**Próximo paso inmediato:** Correr notebooks 01 y 02 en Kaggle GPU para generar los números reales de la tesis.

---

## Dependencias Críticas

```
F0 ──→ F1 (PoC) ──→ comparación final (F7)
              │
F0 ──→ F2 (CNN-Transformer) ──→ F3 (3D Recon) ──→ F4 (Trajectory) ──→ F5 (Robot) ──→ F6 (Execution) ──→ F7 (E2E)
                                                                                                            │
                                                                                                    F8 (tentativa)
                                                                                                            │
                                                                                                    F9 (cierre doc)
```

F1 y F2 pueden paralelizarse parcialmente (F1 primero, F2 empieza en semana 3-4).
F9 empieza en paralelo con F6-F7 (ir generando figuras conforme salen datos).

---

## Regla de Oro

> **Si los phantoms no se hacen, la tesis sigue siendo sólida.**
> El core es Ciencias Computacionales: la arquitectura, el ablation study, la generación de trayectoria conformal, y la validación in-silico.
> Los phantoms son la cereza, no el pastel.
> En ese caso, mover la sección de phantoms a "Future Work" y ajustar la narrativa de Ch.4-6 para reflejar que la validación fue exclusivamente in-silico con 20 modelos sintéticos de geometría conocida.
