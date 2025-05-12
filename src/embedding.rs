use anyhow::Result;
use embed_anything::embeddings::embed::{Embedder as EAEmbedder, TextEmbedder};
use embed_anything::embeddings::local::jina::JinaEmbedder;

pub struct Embedder {
    inner: EAEmbedder,
}

// Default model ID if not specified by the user
// TODO: Explore and evaluate other embedding models (e.g., E5, GTE, BGE) for potentially better performance or domain-specific needs.
// Consider making the embedding model and its dimensions configurable.
const DEFAULT_MODEL_ID: &str = "jinaai/jina-embeddings-v2-small-en";

impl Embedder {
    #[tracing::instrument(fields(model_id = model_id.as_deref().unwrap_or(DEFAULT_MODEL_ID)))]
    pub fn new(
        model_id: Option<String>,
    ) -> Result<Self> {
        let model_to_load = model_id.as_deref().unwrap_or(DEFAULT_MODEL_ID);
        
     
        let jina_embedder = JinaEmbedder::new(model_to_load, None, None).map_err(|e| 
            anyhow::anyhow!("Failed to load JinaEmbedder model '{}': {}. Ensure the model exists and network is available if downloading.", model_to_load, e)
        )?;
        let inner = EAEmbedder::Text(TextEmbedder::Jina(Box::new(jina_embedder)));
        Ok(Self { inner })
    }

    #[tracing::instrument(skip(self, text))]
    pub async fn embed(&self, text: &str) -> Result<[f32; 512]> {
        // This is less efficient, prefer embed_batch for multiple texts
        let mut results = self.embed_batch(&[text], Some(1)).await?;
        if results.is_empty() {
            Err(anyhow::anyhow!("Embedding failed for text"))
        } else {
            Ok(results.remove(0))
        }
    }

    #[tracing::instrument(skip(self, texts))]
    pub async fn embed_batch(&self, texts: &[&str], batch_size: Option<usize>) -> Result<Vec<[f32; 512]>> {
        let results = self.inner.embed(texts, batch_size, None).await?;
        let mut embeddings_array_vec = Vec::with_capacity(results.len());

        for embedding_result in results {
            let vector = embedding_result.to_dense()?; // vector is Vec<f32>
            if vector.len() != 512 { 
                // Assuming Jina v2 models output 512. This might need to be dynamic if other models are used.
                // TODO: The embedding dimension is hardcoded to 512. This should be made dynamic
                // based on the selected embedding model.
                return Err(anyhow::anyhow!(
                    "Embedding size mismatch: expected 512 for Jina v2, got {}",
                    vector.len()
                ));
            }
            let mut arr = [0.0_f32; 512];
            arr.copy_from_slice(&vector);
            embeddings_array_vec.push(arr);
        }
        Ok(embeddings_array_vec)
    }
}