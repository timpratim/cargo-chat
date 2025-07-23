use serde::{Serialize, Deserialize, Serializer, Deserializer};
use serde::ser::SerializeStruct;
use serde::de::{ MapAccess, Error as VError};
use vector::{Index, Vector};
use std::fmt;

pub struct Ann<const D: usize, M> {
    pub index: Index<D>,
    pub vectors: Vec<Vector<D>>,
    pub metadata: Vec<M>,
}

// Dynamic ANN wrapper that can handle different dimensions
pub enum DynamicAnn<M> {
    Dim512(Ann<512, M>),
    Dim1024(Ann<1024, M>),
}

impl<M: Clone> DynamicAnn<M> {
    pub fn build(vectors: Vec<Vec<f32>>, metadata: Vec<M>) -> Result<Self, anyhow::Error> {
        if vectors.is_empty() {
            return Err(anyhow::anyhow!("Cannot build ANN with empty vectors"));
        }
        
        let dimension = vectors[0].len();
        // Validate dimension matches expected values
        if dimension != 512 && dimension != 1024 {
            return Err(anyhow::anyhow!("Unsupported embedding dimension: {}", dimension));
        }
        
        let ann = match dimension {
            512 => {
                let fixed_vectors: Vec<Vector<512>> = vectors
                    .into_iter()
                    .map(|v| {
                        let arr: [f32; 512] = v.try_into()
                            .map_err(|_| anyhow::anyhow!("Failed to convert vector to [f32; 512]"))?;
                        Ok(arr)
                    })
                    .collect::<Result<Vec<_>, anyhow::Error>>()?;
                DynamicAnn::Dim512(Ann::build(&fixed_vectors, &metadata))
            }
            1024 => {
                let fixed_vectors: Vec<Vector<1024>> = vectors
                    .into_iter()
                    .map(|v| {
                        let arr: [f32; 1024] = v.try_into()
                            .map_err(|_| anyhow::anyhow!("Failed to convert vector to [f32; 1024]"))?;
                        Ok(arr)
                    })
                    .collect::<Result<Vec<_>, anyhow::Error>>()?;
                DynamicAnn::Dim1024(Ann::build(&fixed_vectors, &metadata))
            }
            _ => unreachable!("Dimension validation should have caught this")
        };
        
        // Use the dimension method to verify the result
        assert_eq!(ann.dimension(), dimension);
        Ok(ann)
    }

    pub fn query<'a>(&'a self, query: &[f32], k: i32) -> Result<Vec<AnnResult<'a, M>>, anyhow::Error> {
        match self {
            DynamicAnn::Dim512(ann) => {
                let arr: [f32; 512] = query.try_into()
                    .map_err(|_| anyhow::anyhow!("Query vector has wrong dimension for 512D ANN"))?;
                Ok(ann.query(&arr, k))
            }
            DynamicAnn::Dim1024(ann) => {
                let arr: [f32; 1024] = query.try_into()
                    .map_err(|_| anyhow::anyhow!("Query vector has wrong dimension for 1024D ANN"))?;
                Ok(ann.query(&arr, k))
            }
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            DynamicAnn::Dim512(_) => 512,
            DynamicAnn::Dim1024(_) => 1024,
        }
    }
}

impl<M: Serialize + Clone> Serialize for DynamicAnn<M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            DynamicAnn::Dim512(ann) => ann.serialize(serializer),
            DynamicAnn::Dim1024(ann) => ann.serialize(serializer),
        }
    }
}

impl<'de, M: Deserialize<'de> + Clone> Deserialize<'de> for DynamicAnn<M> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Try to deserialize as 512D first, then 1024D
        if let Ok(ann) = Ann::<512, M>::deserialize(deserializer) {
            return Ok(DynamicAnn::Dim512(ann));
        }
        
        // Reset the deserializer and try 1024D
        // This is a bit tricky with serde, so we'll use a different approach
        // For now, we'll require the dimension to be known at compile time
        // and use the original Ann<D, M> for serialization/deserialization
        Err(serde::de::Error::custom("DynamicAnn deserialization not implemented"))
    }
}

impl<const D: usize, M: Serialize> Serialize for Ann<D, M> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Ann", 3)?;
        state.serialize_field("index", &self.index)?;
        let vectors_as_vec: Vec<Vec<f32>> = self.vectors.iter().map(|v| v.to_vec()).collect();
        state.serialize_field("vectors", &vectors_as_vec)?;
        state.serialize_field("metadata", &self.metadata)?;
        state.end()
    }
}

struct AnnVisitor<const D: usize, M> {
    _phantom: std::marker::PhantomData<M>,
}

impl<'de, const D: usize, M: Deserialize<'de>> serde::de::Visitor<'de> for AnnVisitor<D, M> {
    type Value = Ann<D, M>;
    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("struct Ann with metadata")
    }
    fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Self::Value, V::Error> {
        let mut index: Option<Index<D>> = None;
        let mut vectors: Option<Vec<Vector<D>>> = None;
        let mut metadata: Option<Vec<M>> = None;
        while let Some(key) = map.next_key::<&str>()? {
            match key {
                "index" => {
                    if index.is_some() {
                        return Err(VError::duplicate_field("index"));
                    }
                    index = Some(map.next_value()?);
                }
                "vectors" => {
                    if vectors.is_some() {
                        return Err(VError::duplicate_field("vectors"));
                    }
                    let raw_vecs: Vec<Vec<f32>> = map.next_value()?;
                    let mut fixed_vectors = Vec::with_capacity(raw_vecs.len());
                    for v in raw_vecs {
                        if v.len() != D {
                            return Err(VError::custom(format!("Expected vector of length {} but got {}", D, v.len())));
                        }
                        let arr: [f32; D] = v.into_iter().collect::<Vec<_>>().try_into().map_err(|_| VError::custom("Failed to convert Vec<f32> to [f32; D]"))?;
                        fixed_vectors.push(arr);
                    }
                    vectors = Some(fixed_vectors);
                }
                "metadata" => {
                    if metadata.is_some() {
                        return Err(VError::duplicate_field("metadata"));
                    }
                    metadata = Some(map.next_value()?);
                }
                _ => { let _: serde::de::IgnoredAny = map.next_value()?; }
            }
        }
        let index = index.ok_or_else(|| VError::missing_field("index"))?;
        let vectors = vectors.ok_or_else(|| VError::missing_field("vectors"))?;
        let metadata = metadata.ok_or_else(|| VError::missing_field("metadata"))?;
        if vectors.len() != metadata.len() {
            return Err(VError::custom("vectors and metadata length mismatch"));
        }
        Ok(Ann { index, vectors, metadata })
    }
}

impl<'de, const D: usize, M: Deserialize<'de>> Deserialize<'de> for Ann<D, M> {
    fn deserialize<DE: Deserializer<'de>>(deserializer: DE) -> Result<Self, DE::Error> {
        deserializer.deserialize_struct(
            "Ann",
            &["index", "vectors", "metadata"],
            AnnVisitor::<D, M> { _phantom: std::marker::PhantomData },
        )
    }
}

/// Result of an ANN query.
pub struct AnnResult<'a, M> {
    pub metadata: &'a M,
    pub distance: f32,
}

impl<'a, M> AnnResult<'a, M> {
    pub fn new(metadata: &'a M, distance: f32) -> Self {
        Self { metadata, distance }
    }
}

impl<const D: usize, M: Clone> Ann<D, M> {
    pub fn build(vectors: &[Vector<D>], metadata: &[M]) -> Self {
        let index = Index::build(vectors, 16, 100, 42);
        Self {
            index,
            vectors: vectors.to_vec(),
            metadata: metadata.to_vec(),
        }
    }

    pub fn query<'a>(&'a self, query: &Vector<D>, k: i32) -> Vec<AnnResult<'a, M>> {
        let results = self.index.search(&self.vectors, query, k as usize);
        results
            .into_iter()
            .map(|(idx, distance)| AnnResult::new(&self.metadata[idx], distance))
            .collect()
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ChunkMeta {
    pub file: String,
    pub code: String,
}