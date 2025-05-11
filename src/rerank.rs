use anyhow::Result;
use embed_anything::{reranker::model::{Reranker as EAReranker, RerankerResult}, Dtype};
pub struct Reranker {
    inner: EAReranker,
}

impl Reranker {
    pub fn new(_model_dir: &str, _device: ()) -> Result<Self> {
     
        let model_id = "jinaai/jina-reranker-v2-base-multilingual";

        let inner = EAReranker::new(model_id, None, Dtype::F16)?;
        Ok(Self { inner })
    }

    pub fn score(&self, queries: Vec<&str>, documents: Vec<&str>, batch_size: usize) -> Result<Vec<RerankerResult>> {
        let score = self.inner.rerank(queries, documents, batch_size);
        score
    }
}
