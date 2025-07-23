use anyhow::Result;
use embed_anything::embeddings::embed::{Embedder as EAEmbedder, TextEmbedder};
use embed_anything::embeddings::local::jina::JinaEmbedder;
use embed_anything::embeddings::local::qwen3::Qwen3Embedder;
use embed_anything::Dtype;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingModel {
    Jina(String),
    Qwen3(String),
}

impl EmbeddingModel {
    pub fn model_id(&self) -> &str {
        match self {
            EmbeddingModel::Jina(id) => id,
            EmbeddingModel::Qwen3(id) => id,
        }
    }

    pub fn embedding_dimension(&self) -> usize {
        match self {
            EmbeddingModel::Jina(_) => 512,
            EmbeddingModel::Qwen3(_) => 1024,
        }
    }

    pub fn default_jina() -> Self {
        Self::Jina("jinaai/jina-embeddings-v2-small-en".to_string())
    }

    pub fn default_qwen3() -> Self {
        Self::Qwen3("Qwen/Qwen3-Embedding-0.6B".to_string())
    }
}

impl FromStr for EmbeddingModel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_lowercase();
        if s.contains("qwen3") || s.contains("qwen") {
            Ok(EmbeddingModel::Qwen3(s))
        } else if s.contains("jina") {
            Ok(EmbeddingModel::Jina(s))
        } else {
            // Default to Jina for unknown models
            Ok(EmbeddingModel::Jina(s))
        }
    }
}

impl std::fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingModel::Jina(id) => write!(f, "Jina({})", id),
            EmbeddingModel::Qwen3(id) => write!(f, "Qwen3({})", id),
        }
    }
}

pub struct Embedder {
    inner: EAEmbedder,
    model: EmbeddingModel,
}

impl Embedder {
    #[tracing::instrument(fields(model = %model))]
    pub fn new(model: EmbeddingModel) -> Result<Self> {
        let (inner, _) = match &model {
            EmbeddingModel::Qwen3(model_id) => {
                let qwen3_embedder = Qwen3Embedder::new(
                    model_id,
                    None,
                    None,
                    Some(Dtype::F32),
                ).map_err(|e| 
                    anyhow::anyhow!("Failed to load Qwen3Embedder model '{}': {}. Ensure the model exists and network is available if downloading.", model_id, e)
                )?;
                let inner = EAEmbedder::Text(TextEmbedder::Qwen3(Box::new(qwen3_embedder)));
                (inner, 1024)
            }
            EmbeddingModel::Jina(model_id) => {
                let jina_embedder = JinaEmbedder::new(model_id, None, None).map_err(|e| 
                    anyhow::anyhow!("Failed to load JinaEmbedder model '{}': {}. Ensure the model exists and network is available if downloading.", model_id, e)
                )?;
                let inner = EAEmbedder::Text(TextEmbedder::Jina(Box::new(jina_embedder)));
                (inner, 512)
            }
        };
        
        Ok(Self { inner, model })
    }

    /// Get the embedding dimension for this model
    pub fn embedding_dimension(&self) -> usize {
        self.model.embedding_dimension()
    }

    #[tracing::instrument(skip(self, text))]
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // This is less efficient, prefer embed_batch for multiple texts
        let mut results = self.embed_batch(&[text], Some(1)).await?;
        if results.is_empty() {
            Err(anyhow::anyhow!("Embedding failed for text"))
        } else {
            Ok(results.remove(0))
        }
    }

    #[tracing::instrument(skip(self, texts))]
    pub async fn embed_batch(&self, texts: &[&str], batch_size: Option<usize>) -> Result<Vec<Vec<f32>>> {
        let results = self.inner.embed(texts, batch_size, None).await?;
        let mut embeddings_vec = Vec::with_capacity(results.len());

        for embedding_result in results {
            let vector = embedding_result.to_dense()?; // vector is Vec<f32>
            if vector.len() != self.embedding_dimension() { 
                return Err(anyhow::anyhow!(
                    "Embedding size mismatch: expected {} for model, got {}",
                    self.embedding_dimension(),
                    vector.len()
                ));
            }
            embeddings_vec.push(vector);
        }
        Ok(embeddings_vec)
    }
}