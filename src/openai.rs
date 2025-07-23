use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use anyhow::{Result, anyhow};

#[derive(Clone)]
pub struct OpenAIClient {
    pub api_url: String,
    pub api_key: String,
    pub client: Client,
    pub model: String,
}

impl OpenAIClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_url: "https://api.openai.com/v1/chat/completions".to_string(),
            api_key: api_key.into(),
            client: Client::new(),
            model: "gpt-4o".to_string(),
        }
    }

    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

}

#[derive(Serialize, Debug)]
pub struct OpenAIRequest {
  pub model: String,
  pub messages: Vec<Message>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub temperature: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub stream: Option<bool>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub response_format: Option<ResponseFormat>,
}

#[derive(Serialize, Debug)]
pub struct ResponseFormat {
  #[serde(rename = "type")]
  pub format_type: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub json_schema: Option<JsonSchema>,
}

#[derive(Serialize, Debug)]
pub struct JsonSchema {
  pub name: String,
  pub schema: Value,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub strict: Option<bool>,
}

#[derive(Serialize, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIResponse {
    pub choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
pub struct Choice {
    #[serde(default)]
    pub message: ChoiceMessage,
    #[serde(default)]
    pub delta: Option<DeltaChoiceMessage>,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct ChoiceMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct DeltaChoiceMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Deserialize, Debug)]
pub struct StreamChoice {
    pub choices: Vec<Choice>,
} 