use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::{header::HeaderMap, Client};
use serde_json::{json, Value};
use futures_util::{Stream, StreamExt};
use std::pin::Pin;

/// Canonical agent block.
#[derive(Debug, Clone)]
pub enum Block {
    Text(String),
    ToolUse { id: String, name: String, input: Value },
    ToolResult { id: String, name: String, result: Value },
}

/// Canonical message.
#[derive(Debug, Clone)]
pub struct Msg {
    pub role: &'static str,   // "user" | "assistant"
    pub blocks: Vec<Block>,
}

/// Tool spec independent of vendor.
#[derive(Debug, Clone)]
pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    pub schema: Value,        // JSON‑Schema object
}

/// Any LLM back‑end must turn a chat + tool list into blocks.
#[async_trait]
pub trait Backend: Send + Sync {
    async fn chat(&self, chat: &[Msg], tools: &[Tool]) -> Result<Vec<Block>>;
    async fn chat_stream(&self, chat: &[Msg], tools: &[Tool]) -> Result<Pin<Box<dyn Stream<Item = Result<Block>> + Send>>>;
}

fn text_block(t: &str) -> Value {
    json!({ "type": "text", "text": t })
}

fn tool_result_block(id: &str, content: &str) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": id,
        "content": content
    })
}

/* ------------------------------------------------------- */
/*                    Anthropic back‑end                   */
/* ------------------------------------------------------- */

pub struct Anthropic { 
    api_key: String, 
    client: Client,
    model: String,
}

impl Anthropic {
    pub fn new(api_key: String) -> Self {
        Self { 
            api_key, 
            client: Client::new(),
            model: "claude-3-5-sonnet-20241022".to_string(),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

#[async_trait]
impl Backend for Anthropic {
    async fn chat(&self, chat: &[Msg], tools: &[Tool]) -> Result<Vec<Block>> {
        /* 1 – build vendor messages */
        let msgs: Vec<Value> = chat
            .iter()
            .map(|m| {
                let content: Vec<Value> = m
                    .blocks
                    .iter()
                    .map(|b| match b {
                        Block::Text(t) => text_block(t),
                        Block::ToolUse { id, name, input } => json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": input
                        }),
                        Block::ToolResult { id, result, .. } => tool_result_block(id, &result.to_string()),
                    })
                    .collect();
                json!({ "role": m.role, "content": content })
            })
            .collect();

        /* 2 – build vendor tool list */
        let tool_defs: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.schema
                })
            })
            .collect();

        /* 3 – call API */
        let mut hdr = HeaderMap::new();
        hdr.insert("x-api-key", self.api_key.parse()?);
        hdr.insert("anthropic-version", "2023-06-01".parse()?);

        let body = json!({
            "model": self.model,
            "messages": msgs,
            "tools": tool_defs,
            "max_tokens": 2048
        });

        let raw = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .headers(hdr)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        /* 4 – canonicalise */
        parse_anthropic(&raw)
    }

    async fn chat_stream(&self, chat: &[Msg], tools: &[Tool]) -> Result<Pin<Box<dyn Stream<Item = Result<Block>> + Send>>> {
        use async_stream::try_stream;
        
        /* 1 – build vendor messages */
        let msgs: Vec<Value> = chat
            .iter()
            .map(|m| {
                let content: Vec<Value> = m
                    .blocks
                    .iter()
                    .map(|b| match b {
                        Block::Text(t) => text_block(t),
                        Block::ToolUse { id, name, input } => json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": input
                        }),
                        Block::ToolResult { id, result, .. } => tool_result_block(id, &result.to_string()),
                    })
                    .collect();
                json!({ "role": m.role, "content": content })
            })
            .collect();

        /* 2 – build vendor tool list */
        let tool_defs: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.schema
                })
            })
            .collect();

        /* 3 – call API with streaming */
        let mut hdr = HeaderMap::new();
        hdr.insert("x-api-key", self.api_key.parse()?);
        hdr.insert("anthropic-version", "2023-06-01".parse()?);

        let body = json!({
            "model": self.model,
            "messages": msgs,
            "tools": tool_defs,
            "max_tokens": 2048,
            "stream": true
        });

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .headers(hdr)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let stream = try_stream! {
            let mut bs = response.bytes_stream();
            'outer: while let Some(item) = bs.next().await {
                let chunk = item?;
                for line in String::from_utf8_lossy(&chunk).lines() {
                    let line = line.trim();
                    if !line.starts_with("data:") { continue; }
                    let data = line.trim_start_matches("data:").trim();
                    if data == "[DONE]" { break 'outer; }
                    if data.is_empty() { continue; }
                    if let Ok(json_val) = serde_json::from_str::<Value>(data) {
                        if let Some(event_type) = json_val.get("type").and_then(|t| t.as_str()) {
                            tracing::debug!("Anthropic event type: {}", event_type);
                            match event_type {
                                "content_block_delta" => {
                                    if let Some(delta) = json_val.get("delta") {
                                        if let Some(delta_type) = delta.get("type").and_then(|t| t.as_str()) {
                                            if delta_type == "text_delta" {
                                                if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                    if !text.is_empty() {
                                                        yield Block::Text(text.to_string());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                "message_stop" => {
                                    break 'outer;
                                }
                                _ => {
                                    // Ignore other event types
                                }
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

fn parse_anthropic(raw: &Value) -> Result<Vec<Block>> {
    let mut out = Vec::new();
    for item in raw["content"].as_array().ok_or(anyhow!("bad response"))? {
        match item["type"].as_str().unwrap_or("") {
            "text" => out.push(Block::Text(
                item["text"].as_str().unwrap_or("").to_string(),
            )),
            "tool_use" => out.push(Block::ToolUse {
                id: item["id"].as_str().unwrap_or("").to_string(),
                name: item["name"].as_str().unwrap_or("").to_string(),
                input: item["input"].clone(),
            }),
            _ => {}
        }
    }
    Ok(out)
}

/* ------------------------------------------------------- */
/*                      OpenAI back‑end                    */
/* ------------------------------------------------------- */

pub struct OpenAi { 
    api_key: String, 
    client: Client,
    model: String,
    api_url: String,
}

impl OpenAi {
    pub fn new(api_key: String) -> Self {
        Self { 
            api_key, 
            client: Client::new(),
            model: "gpt-4o".to_string(),
            api_url: "https://api.openai.com/v1/chat/completions".to_string(),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    pub fn with_api_url(mut self, url: &str) -> Self {
        self.api_url = url.to_string();
        self
    }
}

#[async_trait]
impl Backend for OpenAi {
    async fn chat(&self, chat: &[Msg], tools: &[Tool]) -> Result<Vec<Block>> {
        /* 1 – build vendor messages */
        let msgs: Vec<Value> = chat
            .iter()
            .flat_map(|m| {
                match m.role {
                    "user" => {
                        vec![json!({
                            "role": "user",
                            "content": m.blocks.iter()
                                .filter_map(|b| match b {
                                    Block::Text(t) => Some(t.clone()),
                                    _ => None
                                })
                                .collect::<Vec<_>>()
                                .join(" ")
                        })]
                    }
                    "assistant" => {
                        let mut msgs = Vec::new();
                        
                        // First, add assistant message with text and tool calls
                        let content = m.blocks.iter()
                            .filter_map(|b| match b {
                                Block::Text(t) => Some(t.clone()),
                                _ => None
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        
                        let tool_calls: Vec<Value> = m.blocks.iter()
                            .filter_map(|b| match b {
                                Block::ToolUse { id, name, input } => Some(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": serde_json::to_string(input).unwrap_or_default()
                                    }
                                })),
                                _ => None
                            })
                            .collect();

                        let mut assistant_msg = json!({
                            "role": "assistant",
                            "content": if content.is_empty() { None } else { Some(content) }
                        });

                        if !tool_calls.is_empty() {
                            assistant_msg["tool_calls"] = tool_calls.into();
                        }

                        msgs.push(assistant_msg);

                        // Then add tool results as separate messages
                        for block in &m.blocks {
                            if let Block::ToolResult { id, result, .. } = block {
                                msgs.push(json!({
                                    "role": "tool",
                                    "tool_call_id": id,
                                    "content": result.to_string()
                                }));
                            }
                        }

                        msgs
                    }
                    _ => vec![]
                }
            })
            .collect();

        /* 2 – functions → tools */
        let tool_defs: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.schema
                    }
                })
            })
            .collect();

        /* 3 – call API */
        let mut hdr = HeaderMap::new();
        hdr.insert("Authorization", format!("Bearer {}", self.api_key).parse()?);

        let mut body = json!({
            "model": self.model,
            "messages": msgs,
            "max_tokens": 2048
        });

        if !tool_defs.is_empty() {
            body["tools"] = tool_defs.into();
        }

        let raw = self
            .client
            .post(&self.api_url)
            .headers(hdr)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<Value>()
            .await?;

        /* 4 – canonicalise */
        parse_openai(&raw)
    }

    async fn chat_stream(&self, chat: &[Msg], tools: &[Tool]) -> Result<Pin<Box<dyn Stream<Item = Result<Block>> + Send>>> {
        use async_stream::try_stream;
        
        /* 1 – build vendor messages */
        let msgs: Vec<Value> = chat
            .iter()
            .flat_map(|m| {
                match m.role {
                    "user" => {
                        vec![json!({
                            "role": "user",
                            "content": m.blocks.iter()
                                .filter_map(|b| match b {
                                    Block::Text(t) => Some(t.clone()),
                                    _ => None
                                })
                                .collect::<Vec<_>>()
                                .join(" ")
                        })]
                    }
                    "assistant" => {
                        let mut msgs = Vec::new();
                        
                        // First, add assistant message with text and tool calls
                        let content = m.blocks.iter()
                            .filter_map(|b| match b {
                                Block::Text(t) => Some(t.clone()),
                                _ => None
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        
                        let tool_calls: Vec<Value> = m.blocks.iter()
                            .filter_map(|b| match b {
                                Block::ToolUse { id, name, input } => Some(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": serde_json::to_string(input).unwrap_or_default()
                                    }
                                })),
                                _ => None
                            })
                            .collect();

                        let mut assistant_msg = json!({
                            "role": "assistant",
                            "content": if content.is_empty() { None } else { Some(content) }
                        });

                        if !tool_calls.is_empty() {
                            assistant_msg["tool_calls"] = tool_calls.into();
                        }

                        msgs.push(assistant_msg);

                        // Then add tool results as separate messages
                        for block in &m.blocks {
                            if let Block::ToolResult { id, result, .. } = block {
                                msgs.push(json!({
                                    "role": "tool",
                                    "tool_call_id": id,
                                    "content": result.to_string()
                                }));
                            }
                        }

                        msgs
                    }
                    _ => vec![]
                }
            })
            .collect();

        /* 2 – functions → tools */
        let tool_defs: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.schema
                    }
                })
            })
            .collect();

        /* 3 – call API with streaming */
        let mut hdr = HeaderMap::new();
        hdr.insert("Authorization", format!("Bearer {}", self.api_key).parse()?);

        let mut body = json!({
            "model": self.model,
            "messages": msgs,
            "max_tokens": 2048,
            "stream": true
        });

        if !tool_defs.is_empty() {
            body["tools"] = tool_defs.into();
        }

        let response = self
            .client
            .post(&self.api_url)
            .headers(hdr)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let stream = try_stream! {
            let mut bs = response.bytes_stream();
            'outer: while let Some(item) = bs.next().await {
                let chunk = item?;
                for line in String::from_utf8_lossy(&chunk).lines() {
                    let line = line.trim();
                    if !line.starts_with("data:") { continue; }
                    let data = line.trim_start_matches("data:").trim();
                    if data == "[DONE]" { break 'outer; }
                    if data.is_empty() { continue; }
                    if let Ok(json_val) = serde_json::from_str::<Value>(data) {
                        tracing::debug!("OpenAI streaming data: {}", data);
                        if let Some(choices) = json_val.get("choices").and_then(|c| c.as_array()) {
                            if let Some(choice) = choices.first() {
                                if let Some(delta) = choice.get("delta") {
                                    // Handle text content
                                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                                        if !content.is_empty() {
                                            yield Block::Text(content.to_string());
                                        }
                                    }
                                    // If this is a tool call, break and let non-streaming handle it
                                    if delta.get("tool_calls").is_some() {
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

fn parse_openai(raw: &Value) -> Result<Vec<Block>> {
    let choice = &raw["choices"][0]["message"];
    let mut out = Vec::new();

    if let Some(c) = choice["content"].as_str() {
        if !c.is_empty() {
            out.push(Block::Text(c.to_string()));
        }
    }

    if let Some(arr) = choice["tool_calls"].as_array() {
        for tc in arr {
            out.push(Block::ToolUse {
                id: tc["id"].as_str().unwrap_or("").to_string(),
                name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                input: serde_json::from_str(
                    tc["function"]["arguments"].as_str().unwrap_or("{}"),
                ).unwrap_or_default(),
            });
        }
    }

    Ok(out)
}
