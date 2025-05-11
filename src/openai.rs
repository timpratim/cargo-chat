use reqwest::Client;
use serde::{Deserialize, Serialize};
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
            model: "gpt-4o-mini".to_string(),
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

#[derive(Serialize)]
pub struct OpenAIRequest {
  pub model: String,
  pub messages: Vec<Message>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_tokens: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub temperature: Option<f32>,
}

#[derive(Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct OpenAIResponse {
    pub choices: Vec<Choice>,
}

#[derive(Deserialize)]
pub struct Choice {
    pub message: ChoiceMessage,
}

#[derive(Deserialize)]
pub struct ChoiceMessage {
    pub content: Option<String>,
} 