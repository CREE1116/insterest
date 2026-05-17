# Insterest Multimodal Recommendation Engine Architecture

This document defines the finalized production architecture of the **Insterest Multimodal Recommendation System**. The system is designed to resolve classical vector collapse and alignment problems by uniting all text projections under a shared weight, avoiding false negatives via SBERT-guided **Soft CLIP**, and establishing a robust baseline for recommendations through **Time-Decay Weighted Pooling**.

---

## 1. Unified Model Structure (UnifiedDiscoveryModel)

To guarantee that search queries, post captions, and hashtags always map to a mathematically consistent latent space, they share a single weight-projection layer (`text_proj`).

```mermaid
graph TD
    subgraph INPUTS["🔤 Inputs"]
        CAP["Caption Text (SBERT 768d)"]
        TAG["Hashtag Text (SBERT 768d)"]
        IMG["Image Pixels (CLIP 512d)"]
    end

    subgraph TEXT_SPACE["🔑 Shared Text Space"]
        TP["text_proj<br/>(MLP 768 ➔ 128)"]
    end

    subgraph IMAGE_SPACE["🖼️ Image Space"]
        IP["image_proj<br/>(MLP 512 ➔ 128)"]
    end

    CAP -->|"SBERT"| TP
    TAG -->|"SBERT"| TP
    IMG -->|"CLIP"| IP

    subgraph COMBINATION["💥 Multimodal Fusion"]
        C_EMB["c_emb (128d, L2 Normed)"]
        H_EMB["h_emb (128d, L2 Normed)"]
        I_EMB["i_emb (128d, L2 Normed)"]
        
        TP --> C_EMB
        TP --> H_EMB
        IP --> I_EMB

        MEAN_TEXT["text_fused (128d)<br/>normalize(c_emb + h_emb)"]
        C_EMB --> MEAN_TEXT
        H_EMB --> MEAN_TEXT

        CONCAT["Concatenated (256d)"]
        MEAN_TEXT --> CONCAT
        I_EMB --> CONCAT

        F_MLP["fusion_mlp<br/>(256 ➔ 128, LayerNorm)"]
        CONCAT --> F_MLP

        FUSED["fused (128d, L2 Normed)"]
        F_MLP --> FUSED

        ADD_IDENT["0.5 × fused + 0.5 × c_emb"]
        FUSED --> ADD_IDENT
        C_EMB --> ADD_IDENT

        ITEM_VEC["Final Item Vector (128d)"]
        ADD_IDENT -->|"L2 Normalize"| ITEM_VEC
    end

    subgraph REDIS["🗄️ Vector Storage"]
        HNSW["Redis HNSW Index"]
        ITEM_VEC --> HNSW
    end
```

---

## 2. Pure Semantic Search (100% Query Search)

By completely bypassing the dynamic gating fusion loop when a user submits a search query, we ensure that searches are purely focused on semantic and thematic matches without any personalization bias distortion.

```mermaid
sequenceDiagram
    autonumber
    actor User as Search Client
    participant App as Discovery API / intelligence.py
    participant SBERT as SBERT Embedder
    participant Model as UnifiedDiscoveryModel
    participant Redis as Redis ANN Vector DB

    User->>App: GET /api/v1/discovery?query_text="고양이"
    App->>SBERT: get_raw_query_vector("고양이")
    SBERT-->>App: 768d Dense Vector
    App->>Model: get_query_embedding(768d)
    Model->>Model: text_proj(768d) & L2-norm
    Model-->>App: 128d Unified Query Embedding
    App->>Redis: Search KNN (HNSW index, limit=Top-K)
    Redis-->>App: List of matching post IDs & Similarity Scores
    App-->>User: JSON Response (Posts containing cats / similar visual motifs)
```

---

## 3. Personalized Recommendation Flow (Time-Decay Weighted Pooling)

Rather than using an untrained, sequence-distorting GRU model when likes data is sparse, the system maps user preferences using **Time-Decay Weighted Pooling**. The older a like interaction, the exponentially smaller its impact on the resulting user vector.

```mermaid
flowchart TD
    subgraph USER_HISTORY["🕒 Interaction History (ASC)"]
        L1["Like 1 (Oldest)"]
        L2["Like 2"]
        L3["Like 3"]
        LN["Like N (Most Recent)"]
    end

    subgraph EMBEDDINGS["📦 Item Representations"]
        E1["Item 1 (128d)"]
        E2["Item 2 (128d)"]
        E3["Item 3 (128d)"]
        EN["Item N (128d)"]
    end

    L1 --> E1
    L2 --> E2
    L3 --> E3
    LN --> EN

    subgraph DECAY["⏳ Exponential Time-Decay Weights"]
        W1["Weight = 0.5^(N-1)"]
        W2["Weight = 0.5^(N-2)"]
        W3["Weight = 0.5^(N-3)"]
        WN["Weight = 0.5^0 = 1.0 (Max weight)"]
    end

    E1 -->|"Multiply"| W1
    E2 -->|"Multiply"| W2
    E3 -->|"Multiply"| W3
    EN -->|"Multiply"| WN

    W1 & W2 & W3 & WN -->|"Sum & Normalize (L2)"| USER_VEC["Centroid User Vector (128d)"]

    USER_VEC -->|"discovery(zeros, user_vec)"| GATING["Dynamic Gating Gated User = user_vec"]
    GATING -->|"Redis ANN Search"| REDIS["Redis DB"]
    REDIS --> FEED["Personalized Taste Feed"]
```

---

## 4. Phase 1 Training Pipeline: SBERT-Guided Bidirectional Soft CLIP

To align images and texts while preventing false negatives (e.g., classifying "Cat 1" and "Cat 2" in a batch as total opposites), the alignment loss is calculated against soft probabilities derived from SBERT similarity scores.

```mermaid
flowchart TD
    subgraph IN["📖 Input Batches"]
        C_RAW["Caption Vectors [B, 768] (SBERT)"]
        I_RAW["Image Vectors [B, 512] (CLIP)"]
    end

    subgraph TEACHER["🧑‍🏫 Teacher: SBERT Semantics"]
        T_NORM["Normalize raw SBERT"]
        T_SIM["teacher_sim = c_raw @ c_raw.T"]
        SOFT_PROB["soft_labels = Softmax(teacher_sim / 0.07)"]
        
        T_NORM --> T_SIM
        T_SIM --> SOFT_PROB
    end

    C_RAW --> TEACHER

    subgraph STUDENT["👶 Student: Model Projection"]
        C_PROJ["c_emb = Normalize(text_proj(c_raw))"]
        I_PROJ["i_emb = Normalize(image_proj(i_raw))"]

        L_SP["similarity_preservation_loss<br/>MSE(student_sim, teacher_sim)"]
        
        CROSS_LOGITS["logits_i2t = i_emb @ c_emb.T<br/>logits_t2i = c_emb @ i_emb.T"]
        
        L_I2T["loss_i2t = KL(logits_i2t, soft_labels)<br/>(Updates image_proj)"]
        L_T2I["loss_t2i = KL(logits_t2i, soft_labels)<br/>(Updates text_proj lightly)"]
    end

    C_RAW --> C_PROJ
    I_RAW --> I_PROJ

    C_PROJ --> L_SP
    C_PROJ & I_PROJ --> CROSS_LOGITS
    SOFT_PROB --> L_I2T & L_T2I
    CROSS_LOGITS --> L_I2T & L_T2I

    subgraph OPTIMIZER["📈 Multi-Loss Optimization"]
        TOTAL_LOSS["total_loss = loss_struct + loss_i2t + 0.2 × loss_t2i"]
        TOTAL_LOSS -->|"Backward & Step"| UPDATE["Update text_proj, image_proj, and fusion_mlp"]
    end

    L_SP & L_I2T & L_T2I --> TOTAL_LOSS
```

---

## 5. Architectural Comparison (Before vs. After Optimization)

| Feature | Before (Collapsed Space) | After (Aligned Stable Space) | Rationale |
|:---|:---|:---|:---|
| **Text Embedding Projection** | Separated `query_proj` & `caption_proj` | **Unified `text_proj`** | Guarantees search terms and post descriptions share the exact same coordinates by design. |
| **Hashtag Processing** | Separated `hashtag_proj` (untrained noise) | **Shared `text_proj` (averaged)** | Merges hashtag signals directly with caption semantics as a regularized text input. |
| **Image-Text Alignment** | Hard CLIP-style InfoNCE Loss | **SBERT-Guided Bidirectional Soft CLIP** | Solves False Negatives. batched duplicates (e.g., Cat 1 & Cat 2) do not push each other away. |
| **Alignment Gradient Path** | Bidirectional (Distorted text weights) | **Scaled Bidirectional (0.2 weight on T2I)** | Prevents visual details from distorting SBERT's highly consistent textual meaning structure. |
| **Personalization Model** | Untrained random GRU/Attention UserTower | **Exponential Time-Decay Weighted Pooling** | Represents sequential interests perfectly (weighting recent items double) without training complexity. |
| **Search Engine Fusion** | Dynamic Gating (10% user bias in search) | **Pure Query Search Bypass** | Preserves 100% search intent precision by removing personalization noise from discoveries. |
