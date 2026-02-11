# A Technical Plan Based on Causal Inference and Dynamic Network Discovery (Preliminary Concept)

## 1. Background

The integrated traditional Chinese and Western medicine diagnosis and treatment of nasopharyngeal carcinoma (NPC) has accumulated massive amounts of data. However, “efficacy evaluation” has long remained at the level of correlation analysis (Correlation). What we want to answer is the question: “After taking this herbal medicine, will the patient’s condition improve?”


This project utilizes a unique dataset containing **963 records** and plans to adopt a **Causal Inference** framework to strip away confounding factors, precisely quantify the true therapeutic effects of Chinese herbal medicine on NPC rehabilitation patients, and uncover potential diagnostic and therapeutic mechanisms.

---

## 2. Research Challenges

### 2.1 Challenge One: Lack of Causal “Ground Truth”

**Problem**: Due to the absence of a god’s-eye view, we do not have a true causal graph. The causal graph recovered by algorithms may be spurious statistical associations.  
**Solutions**:

1. **Expert Knowledge**: Transform traditional Chinese medicine theory into algorithmic **hard constraints (Hard Constraints)** or **prior probabilities (Priors)**.  
* *Operation*: Enforce that causal relationships must consider temporal factors; strictly prohibit biologically impossible edges; incorporate the meridian attribution of herbal effects from *Chinese Materia Medica* as prior knowledge. -- This idea is yet to be implemented; such requirements are relatively common in other papers.

2. **Expert Validation**: Set up blind evaluations. Present the recovered causal relationships to experts for judgment on whether they truly conform to causality. -- This requires time, and the workload for physicians is substantial.

### 2.2 Challenge Two: Instability of Results

**Problem**: With a small sample size (963 records), removing a few data points may lead to changes in the causal graph structure, or even result in completely different causal graphs each time.  
**Solutions**:

1. **Stability Selection**: Do not rely on a single one-time training result.  
* *Operation*: Perform 1000 rounds of Bootstrap resampling on the original data. Run the causal discovery algorithm on each resampled dataset.  
* *Threshold Filtering*: Compute the frequency of occurrence (Selection Probability) for each edge. Retain only those edges that appear in more than 85% of the resamples. This can greatly reduce the false positive rate.

## Objectives

### 3.1 Objective One: Confirmatory Causal Relationships

**Core Task**: **Simulate** randomized controlled trials (RCTs). For specific hypotheses (e.g., “Do qi-tonifying and blood-activating herbs improve fatigue?”), calculate the Average Treatment Effect (ATE).

#### Step 1: Define Treatment Variables and Outcome Variables 

* The original data are complex text and must be binarized or numerically encoded before entering causal models.

* **Treatment Group**: Based on the `chinese_medicines` field, if the prescription contains a specific category of herbs (e.g., “Astragalus + Salvia” or “qi-tonifying intensity > threshold”), label it as 1.
* **Control Group**: Patients whose prescriptions do not contain this category of herbs, or contain other categories.
* **Outcome**: Calculate the improvement of the target symptom (e.g., “fatigue”) in the `chief_complaint` at time $t+1$.

#### Step 2: Construct the Confounder Set 

* In observational studies, patients who take medication are often those with more severe conditions. If confounders are not controlled, the therapeutic effect of drugs will be masked by disease severity (Simpson’s paradox).
* Use domain knowledge to select variables $Z$ that must be controlled:
* **Demographics**: `age`, `gender`.
* **TCM Syndrome Differentiation**: `chinese_diagnosis` is the most important confounder (because physicians prescribe herbs based on the “syndrome”). Syndrome text needs to be converted into multi-hot encoding.
* **Baseline Condition**: Symptom severity at time .
* **Western Medical Treatment**: `western_medicines` (the history of radiotherapy and chemotherapy has a substantial impact on symptoms).

#### Step 3: Propensity Score Matching

* To make the treatment and control groups as similar as possible in terms of baseline feature **distributions**, thereby simulating random assignment.
* **Score Calculation**: Use confounders $Z$ as input to predict the probability of a patient receiving treatment $T$ (Propensity Score).
* **Matching/Weighting**: Pair patients with similar scores, or use inverse probability weighting (IPW).    
* **Balance Check**: After matching, examine whether there are no significant differences between the two groups in the distributions of `age`, `chinese_diagnosis`, etc.

Note: My understanding of “propensity score matching” is still insufficient, but I have seen other papers use this method to “simulate random assignment.” I believe this method is feasible, though further study is required.

#### Step 4: Causal Effect Estimation

* On the basis of balanced data, compute the net effect of the drug.
* Compare the symptom improvement rates of the two matched groups at time $t+1$.
* Formula: $ATE = E[Y|do(T=1)] - E[Y|do(T=0)]$.
* If the result is significantly positive, it statistically verifies that “the drug is effective.”

---

### 3.2 Objective Two: Causal Discovery 

**Core Task**: Construct a causal network. Utilize temporal ordering to mine the evolutionary pathways among “symptoms–medications–syndromes” from a large number of variables.

#### Step 1: Construct a Temporal Data Matrix

* **Why**: Causal discovery algorithms typically handle static data. We need to transform the data into a “time-slice” format to reflect the influence of time $t$ on time $t+1$.
* For each patient’s visit sequence, construct lagged features.
* **Row Structure**: Each row contains `[symptoms_t, medications_t, syndromes_t]`.
* **Missing Data Handling**: Due to irregular follow-ups, sample pairs with overly long time intervals (e.g., > 6 months) need to be removed to avoid breaking causal chains.

#### Step 2: Variable Dimensionality Reduction and Clustering

* There are hundreds of herbal medicines and numerous symptom descriptions. Directly placing hundreds of variables into a causal graph would cause computational explosion and poor interpretability.
* **Herbal Classification**: Use knowledge from *Chinese Materia Medica* to cluster `chinese_medicines` into several major categories (e.g., qi-tonifying, yin-nourishing, blood-activating, heat-clearing and detoxifying, etc.).
* **Symptom Classification**: Cluster entities extracted from `chief_complaint` into core symptom groups (e.g., ear–nose symptoms, systemic symptoms, sleep and digestion).
* Control the final number of nodes in the model to within 20–50.

#### Step 3: Constraint-Based Structure Learning

* Use algorithms to automatically find conditional independencies among variables and construct a directed acyclic graph (DAG).
* **Apply Hard Constraints (Key Step)**:
1. **Prohibit Reverse Edges**: Variables at time $t+1$ must never point to variables at time $t$ (the future cannot change the past).
2. **Prohibit Instantaneous Edges**: Assume that drugs take time to exert effects; prohibit `medications_t -> symptoms_t`, and instead search for `medications_t -> symptoms_{t+1}`.

* **Run Algorithms**: Use the PC algorithm or the FCI algorithm (considering the possible existence of unobserved latent variables) for initial graph construction.

Note: We have already encountered and used both algorithms. The difficulty lies in the uncertainty of the recovery quality, and even the fact that results may differ each time the algorithm is run. Multiple runs may be required here.

#### Step 4: Stability Filtering and Expert Revision 

*Small data size (963 records) can lead to graphs containing a large number of spurious correlations.

* **Bootstrap Resampling**: Following the method in Step 3, run 1000 times and count the frequency of each edge.
* **Retain High-Confidence Edges**: Keep only edges with occurrence frequency > 85%.
* **Knowledge Pruning**: Invite TCM experts to review the remaining edges and remove paths that violate medical common sense (e.g., `qi-tonifying herbs -> cause qi deficiency`, which may be a false correlation caused by disease progression and should be removed).

#### Step 5: Path Interpretation and Visualization

* Translate complex mathematical graphs into diagnostic and therapeutic mechanisms that physicians can understand.
* Extract specific subgraph paths, for example: `qi deficiency syndrome(t) -> qi-tonifying herbs(t) -> fatigue improvement(t+1)`.

## 4. Data Preprocessing

#### 1. Timeline Reconstruction and Window Alignment

Because the `time` field shows irregular visit intervals (ranging from 1 week to several months), direct modeling would introduce bias.

* **Relative Timeline**: Set each patient’s (`patient_id`) first visit time as `t_0`.
* **Time Window Binning**: Define a standard time step (e.g., 1 month).
* If multiple records exist within the same window (e.g., patients with frequent short-term follow-ups as in Example 1), adopt an **aggregation strategy**: take the maximum severity for symptoms, and take the union or intersection for medications.

#### 2. Feature Engineering for Unstructured Text 

For `chief_complaint` (chief complaint) and `chinese_medicines` (prescriptions):

* **A. Symptom Quantification (Symptom Quantification)**:
* **Entity Extraction**: Build a keyword dictionary (e.g., headache, nasal congestion, fatigue, poor sleep).
* **Severity Determination**: Use regular expressions to match modifying words.
* “No headache”, “headache improved”  0 (none/mild)
* “Headache”, “occasional headache”  1 (moderate)
* “Obvious headache”, “severe headache”  2 (severe)

* **Output**: Generate structured vectors.

Note: LLMs can be directly used to implement this.

* **B. Herbal Medicine Dimensionality Reduction (Herbal Medicine Mapping)**:
* **Mapping Logic**: Establish a mapping table of `herb -> efficacy category`.
* *Input*: `[\"Astragalus\", \"Atractylodes\", \"Salvia\"]`
* *Mapping*: Astragalus/Atractylodes  qi-tonifying herbs; Salvia  blood-activating herbs.

* **Vectorization**: Calculate efficacy intensity.
* *Output*: `[qi-tonifying intensity: 2, blood-activating intensity: 1, heat-clearing intensity: 0, ...]`
