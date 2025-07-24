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
        
        if metadata.is_empty() {
            return Err(anyhow::anyhow!("Cannot build ANN with empty metadata"));
        }
        
        if vectors.len() != metadata.len() {
            return Err(anyhow::anyhow!("Vector count ({}) does not match metadata count ({})", 
                vectors.len(), metadata.len()));
        }
        
        let dimension = vectors[0].len();
        if dimension == 0 {
            return Err(anyhow::anyhow!("Cannot build ANN with zero-dimensional vectors"));
        }
        
        // Validate all vectors have the same dimension
        for (i, vector) in vectors.iter().enumerate() {
            if vector.len() != dimension {
                return Err(anyhow::anyhow!("Vector {} has dimension {} but expected {}", 
                    i, vector.len(), dimension));
            }
        }
        
        // Validate dimension matches expected values
        if dimension != 512 && dimension != 1024 {
            return Err(anyhow::anyhow!("Unsupported embedding dimension: {}. Only 512 and 1024 are supported.", dimension));
        }
        
        let ann = match dimension {
            512 => {
                tracing::debug!("Building 512-dimensional ANN with {} vectors", vectors.len());
                let fixed_vectors: Vec<Vector<512>> = vectors
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let arr: [f32; 512] = v.try_into()
                            .map_err(|_| anyhow::anyhow!("Failed to convert vector {} to [f32; 512]: length mismatch", i))?;
                        Ok(arr)
                    })
                    .collect::<Result<Vec<_>, anyhow::Error>>()?;
                DynamicAnn::Dim512(Ann::build(&fixed_vectors, &metadata))
            }
            1024 => {
                tracing::debug!("Building 1024-dimensional ANN with {} vectors", vectors.len());
                let fixed_vectors: Vec<Vector<1024>> = vectors
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let arr: [f32; 1024] = v.try_into()
                            .map_err(|_| anyhow::anyhow!("Failed to convert vector {} to [f32; 1024]: length mismatch", i))?;
                        Ok(arr)
                    })
                    .collect::<Result<Vec<_>, anyhow::Error>>()?;
                DynamicAnn::Dim1024(Ann::build(&fixed_vectors, &metadata))
            }
            _ => unreachable!("Dimension validation should have caught this")
        };
        
        // Verify the result
        let actual_dimension = ann.dimension();
        if actual_dimension != dimension {
            return Err(anyhow::anyhow!("ANN dimension mismatch: expected {}, got {}", dimension, actual_dimension));
        }
        
        tracing::info!("Successfully built {}-dimensional ANN with {} vectors", dimension, metadata.len());
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

impl<'de, M> Deserialize<'de> for DynamicAnn<M> 
where 
    M: serde::de::DeserializeOwned + Clone,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // We need to deserialize to a generic representation first, then determine the dimension
        let value: serde_json::Value = serde_json::Value::deserialize(deserializer)?;
        
        // Extract the vectors to determine dimension
        let vectors = value.get("vectors")
            .and_then(|v| v.as_array())
            .ok_or_else(|| serde::de::Error::custom("Missing or invalid 'vectors' field"))?;
        
        if vectors.is_empty() {
            return Err(serde::de::Error::custom("Cannot deserialize DynamicAnn with empty vectors"));
        }
        
        // Check the dimension of the first vector
        let first_vector = vectors[0].as_array()
            .ok_or_else(|| serde::de::Error::custom("Invalid vector format"))?;
        
        let dimension = first_vector.len();
        
        // Deserialize based on detected dimension
        match dimension {
            512 => {
                let ann: Ann<512, M> = serde_json::from_value(value)
                    .map_err(|e| serde::de::Error::custom(format!("Failed to deserialize as 512D: {}", e)))?;
                Ok(DynamicAnn::Dim512(ann))
            },
            1024 => {
                let ann: Ann<1024, M> = serde_json::from_value(value)
                    .map_err(|e| serde::de::Error::custom(format!("Failed to deserialize as 1024D: {}", e)))?;
                Ok(DynamicAnn::Dim1024(ann))
            },
            _ => Err(serde::de::Error::custom(format!("Unsupported dimension: {}. Only 512 and 1024 are supported.", dimension)))
        }
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
    pub language: Option<String>,
    pub extension: Option<String>,
}