use anyhow::Result;
use embed_anything::embeddings::embed::{Embedder as EAEmbedder, TextEmbedder};
use embed_anything::embeddings::local::jina::JinaEmbedder;

pub struct Embedder {
    inner: EAEmbedder,
}

impl Embedder {
    pub fn new(_model_dir: Option<String>, _device: ()) -> Result<Self> {
        // For now, always use Jina local embedder. Extend as needed.
        let inner = EAEmbedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())));
        Ok(Self { inner })
    }

    pub async fn embed(&self, text: &str) -> Result<[f32; 512]> {
        let results = self.inner.embed(&[text], None, None).await?;
        let vector = results[0].to_dense()?; // Now vector is Vec<f32>
        
        if vector.len() != 512 {
            return Err(anyhow::anyhow!(
                "Embedding size mismatch: expected 512, got {}",
                vector.len()
            ));
        }
        
        let mut arr = [0.0_f32; 512];
        arr.copy_from_slice(&vector);
        Ok(arr)
    }
}
