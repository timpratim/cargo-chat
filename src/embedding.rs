use anyhow::Result;
use embed_anything::embeddings::embed::{Embedder as EAEmbedder, TextEmbedder};
use embed_anything::embeddings::local::jina::JinaEmbedder;

pub struct Embedder {
    inner: EAEmbedder,
}

impl Embedder {
    #[tracing::instrument]
    pub fn new(_model_dir: Option<String>, _device: ()) -> Result<Self> {
        // For now, always use Jina local embedder. Extend as needed.
        let inner = EAEmbedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())));
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
                return Err(anyhow::anyhow!(
                    "Embedding size mismatch: expected 512, got {}",
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