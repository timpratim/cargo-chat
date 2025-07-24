use crate::language::{detect_language_from_extension, SupportedLanguage};
use anyhow::Result;
use log::{debug, info, warn};
use std::collections::HashMap;
use std::path::Path;
use walkdir::WalkDir;

/// Repository profile containing metadata about a codebase
#[derive(Debug, Clone)]
pub struct RepoProfile {
    /// Repository name (derived from root directory name)
    pub name: String,
    /// Primary programming languages ordered by lines of code
    pub primary_languages: Vec<String>,
    /// Build/configuration files found in the repository
    pub build_files: Vec<String>,
    /// Detected frameworks and libraries
    pub frameworks: Vec<String>,
    /// Optional 1-2 sentence summary of README content
    pub readme_summary: Option<String>,
}

impl RepoProfile {
    /// Create a repository profile by analyzing the given directory
    pub async fn from_directory(path: &Path) -> Result<RepoProfile> {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown-repo")
            .to_string();

        info!("Profiling repository: {}", name);

        let (language_stats, build_files) = Self::analyze_files(path)?;
        let primary_languages = Self::get_primary_languages(&language_stats);
        let frameworks = Self::detect_frameworks(&build_files, path).await?;
        let readme_summary = Self::summarize_readme(path).await;

        Ok(RepoProfile {
            name,
            primary_languages,
            build_files,
            frameworks,
            readme_summary,
        })
    }

    /// Analyze files in the repository to gather statistics
    fn analyze_files(path: &Path) -> Result<(HashMap<SupportedLanguage, usize>, Vec<String>)> {
        let mut language_stats: HashMap<SupportedLanguage, usize> = HashMap::new();
        let mut build_files = Vec::new();

        for entry in WalkDir::new(path)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let file_path = entry.path();
            if !file_path.is_file() {
                continue;
            }

            let file_name = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // Check for build/config files
            if Self::is_build_file(file_name) {
                build_files.push(file_name.to_string());
                continue;
            }

            // Count lines by language
            if let Some(extension) = file_path.extension().and_then(|e| e.to_str()) {
                if let Some(language) = detect_language_from_extension(extension) {
                    if let Ok(content) = std::fs::read_to_string(file_path) {
                        let line_count = content.lines().count();
                        *language_stats.entry(language).or_insert(0) += line_count;
                    }
                }
            }
        }

        Ok((language_stats, build_files))
    }

    /// Check if a filename indicates a build or configuration file
    fn is_build_file(filename: &str) -> bool {
        matches!(
            filename.to_lowercase().as_str(),
            "cargo.toml"
                | "cargo.lock"
                | "package.json"
                | "package-lock.json"
                | "yarn.lock"
                | "pom.xml"
                | "build.gradle"
                | "gradle.properties"
                | "makefile"
                | "cmake.txt"
                | "cmakelists.txt"
                | "pyproject.toml"
                | "poetry.lock"
                | "requirements.txt"
                | "setup.py"
                | "gemfile"
                | "gemfile.lock"
                | "go.mod"
                | "go.sum"
                | "build.rs"
                | "build.zig"
                | "dune-project"
                | "dune"
                | "mix.exs"
                | "rebar.config"
                | "stack.yaml"
                | "cabal.project"
                | "project.clj"
                | "deps.edn"
                | "composer.json"
                | "composer.lock"
        )
    }

    /// Extract primary languages from language statistics
    fn get_primary_languages(language_stats: &HashMap<SupportedLanguage, usize>) -> Vec<String> {
        let mut languages: Vec<(SupportedLanguage, usize)> = language_stats
            .iter()
            .map(|(lang, &count)| (lang.clone(), count))
            .collect();

        // Sort by line count descending
        languages.sort_by(|a, b| b.1.cmp(&a.1));

        // Take top 5 languages with meaningful presence
        languages
            .into_iter()
            .take(5)
            .filter(|(_, count)| *count > 10) // Only include languages with >10 lines
            .map(|(lang, _)| lang.display_name().to_string())
            .collect()
    }

    /// Detect frameworks and libraries from build files and file analysis
    async fn detect_frameworks(build_files: &[String], repo_path: &Path) -> Result<Vec<String>> {
        let mut frameworks = Vec::new();

        for build_file in build_files {
            match build_file.as_str() {
                "Cargo.toml" => frameworks.extend(Self::detect_rust_frameworks(repo_path)),
                "package.json" => frameworks.extend(Self::detect_js_frameworks(repo_path)),
                "pom.xml" => frameworks.extend(Self::detect_java_frameworks(repo_path)),
                "requirements.txt" | "pyproject.toml" => {
                    frameworks.extend(Self::detect_python_frameworks(repo_path))
                }
                "go.mod" => frameworks.extend(Self::detect_go_frameworks(repo_path)),
                _ => {}
            }
        }

        // Remove duplicates and limit to top frameworks
        frameworks.sort();
        frameworks.dedup();
        frameworks.truncate(10);

        Ok(frameworks)
    }

    /// Detect Rust frameworks from Cargo.toml
    fn detect_rust_frameworks(repo_path: &Path) -> Vec<String> {
        let cargo_path = repo_path.join("Cargo.toml");
        if let Ok(content) = std::fs::read_to_string(cargo_path) {
            let mut frameworks = Vec::new();

            if content.contains("tokio") {
                frameworks.push("Tokio".to_string());
            }
            if content.contains("actix") {
                frameworks.push("Actix".to_string());
            }
            if content.contains("axum") {
                frameworks.push("Axum".to_string());
            }
            if content.contains("warp") {
                frameworks.push("Warp".to_string());
            }
            if content.contains("rocket") {
                frameworks.push("Rocket".to_string());
            }
            if content.contains("serde") {
                frameworks.push("Serde".to_string());
            }
            if content.contains("clap") {
                frameworks.push("Clap".to_string());
            }
            if content.contains("diesel") {
                frameworks.push("Diesel".to_string());
            }
            if content.contains("sqlx") {
                frameworks.push("SQLx".to_string());
            }
            if content.contains("bevy") {
                frameworks.push("Bevy".to_string());
            }
            if content.contains("tauri") {
                frameworks.push("Tauri".to_string());
            }

            frameworks
        } else {
            Vec::new()
        }
    }

    /// Detect JavaScript/TypeScript frameworks from package.json
    fn detect_js_frameworks(repo_path: &Path) -> Vec<String> {
        let package_path = repo_path.join("package.json");
        if let Ok(content) = std::fs::read_to_string(package_path) {
            let mut frameworks = Vec::new();

            if content.contains("react") {
                frameworks.push("React".to_string());
            }
            if content.contains("vue") {
                frameworks.push("Vue".to_string());
            }
            if content.contains("angular") {
                frameworks.push("Angular".to_string());
            }
            if content.contains("svelte") {
                frameworks.push("Svelte".to_string());
            }
            if content.contains("next") {
                frameworks.push("Next.js".to_string());
            }
            if content.contains("nuxt") {
                frameworks.push("Nuxt.js".to_string());
            }
            if content.contains("express") {
                frameworks.push("Express".to_string());
            }
            if content.contains("fastify") {
                frameworks.push("Fastify".to_string());
            }
            if content.contains("nestjs") {
                frameworks.push("NestJS".to_string());
            }
            if content.contains("typescript") {
                frameworks.push("TypeScript".to_string());
            }
            if content.contains("webpack") {
                frameworks.push("Webpack".to_string());
            }
            if content.contains("vite") {
                frameworks.push("Vite".to_string());
            }

            frameworks
        } else {
            Vec::new()
        }
    }

    /// Detect Java frameworks from pom.xml
    fn detect_java_frameworks(repo_path: &Path) -> Vec<String> {
        let pom_path = repo_path.join("pom.xml");
        if let Ok(content) = std::fs::read_to_string(pom_path) {
            let mut frameworks = Vec::new();

            if content.contains("spring") {
                frameworks.push("Spring".to_string());
            }
            if content.contains("junit") {
                frameworks.push("JUnit".to_string());
            }
            if content.contains("hibernate") {
                frameworks.push("Hibernate".to_string());
            }
            if content.contains("jackson") {
                frameworks.push("Jackson".to_string());
            }
            if content.contains("maven") {
                frameworks.push("Maven".to_string());
            }

            frameworks
        } else {
            Vec::new()
        }
    }

    /// Detect Python frameworks from requirements.txt or pyproject.toml
    fn detect_python_frameworks(repo_path: &Path) -> Vec<String> {
        let mut frameworks = Vec::new();

        // Check requirements.txt
        let req_path = repo_path.join("requirements.txt");
        if let Ok(content) = std::fs::read_to_string(req_path) {
            if content.contains("django") {
                frameworks.push("Django".to_string());
            }
            if content.contains("flask") {
                frameworks.push("Flask".to_string());
            }
            if content.contains("fastapi") {
                frameworks.push("FastAPI".to_string());
            }
            if content.contains("numpy") {
                frameworks.push("NumPy".to_string());
            }
            if content.contains("pandas") {
                frameworks.push("Pandas".to_string());
            }
            if content.contains("pytorch") {
                frameworks.push("PyTorch".to_string());
            }
            if content.contains("tensorflow") {
                frameworks.push("TensorFlow".to_string());
            }
        }

        // Check pyproject.toml
        let pyproject_path = repo_path.join("pyproject.toml");
        if let Ok(content) = std::fs::read_to_string(pyproject_path) {
            if content.contains("poetry") {
                frameworks.push("Poetry".to_string());
            }
        }

        frameworks
    }

    /// Detect Go frameworks from go.mod
    fn detect_go_frameworks(repo_path: &Path) -> Vec<String> {
        let go_mod_path = repo_path.join("go.mod");
        if let Ok(content) = std::fs::read_to_string(go_mod_path) {
            let mut frameworks = Vec::new();

            if content.contains("gin") {
                frameworks.push("Gin".to_string());
            }
            if content.contains("echo") {
                frameworks.push("Echo".to_string());
            }
            if content.contains("fiber") {
                frameworks.push("Fiber".to_string());
            }
            if content.contains("gorm") {
                frameworks.push("GORM".to_string());
            }
            if content.contains("cobra") {
                frameworks.push("Cobra".to_string());
            }

            frameworks
        } else {
            Vec::new()
        }
    }

    /// Attempt to generate a brief summary of README content
    /// For now, returns None since we don't have LLM integration here yet
    /// This can be enhanced later to use the OpenAI client for summarization
    async fn summarize_readme(repo_path: &Path) -> Option<String> {
        let readme_candidates = ["README.md", "README.txt", "README"];

        for candidate in &readme_candidates {
            let readme_path = repo_path.join(candidate);
            if let Ok(content) = std::fs::read_to_string(readme_path) {
                // For now, just extract first paragraph or first 200 chars
                let summary = content
                    .lines()
                    .take(5) // Take first 5 lines
                    .collect::<Vec<_>>()
                    .join(" ")
                    .chars()
                    .take(200) // Limit to 200 characters
                    .collect::<String>();

                if !summary.trim().is_empty() {
                    return Some(format!("{}...", summary.trim()));
                }
            }
        }
        None
    }

    /// Format a list of items into human-readable text with proper conjuction
    pub fn human_list(items: &[String], conjunction: &str) -> String {
        match items.len() {
            0 => String::new(),
            1 => items[0].clone(),
            2 => format!("{} {} {}", items[0], conjunction, items[1]),
            _ => {
                let (last, rest) = items.split_last().unwrap();
                format!("{}, {} {}", rest.join(", "), conjunction, last)
            }
        }
    }

    /// Generate a human-readable description of the repository
    pub fn description(&self) -> String {
        let mut parts = Vec::new();

        // Add language info
        if !self.primary_languages.is_empty() {
            let lang_desc = if self.primary_languages.len() == 1 {
                format!("a {} codebase", self.primary_languages[0])
            } else {
                format!(
                    "a multi-language codebase primarily using {}",
                    Self::human_list(&self.primary_languages, "and")
                )
            };
            parts.push(lang_desc);
        }

        // Add framework info
        if !self.frameworks.is_empty() {
            let framework_desc =
                format!("built with {}", Self::human_list(&self.frameworks, "and"));
            parts.push(framework_desc);
        }

        // Combine parts
        if parts.is_empty() {
            format!("the '{}' repository", self.name)
        } else {
            format!("'{}', {}", self.name, parts.join(" "))
        }
    }

    /// Get the project type based on build files and languages
    pub fn project_type(&self) -> String {
        // Check build files first for specific project types
        for build_file in &self.build_files {
            match build_file.as_str() {
                "Cargo.toml" => return "Rust project".to_string(),
                "package.json" => return "Node.js project".to_string(),
                "pom.xml" => return "Maven Java project".to_string(),
                "build.gradle" => return "Gradle project".to_string(),
                "go.mod" => return "Go module".to_string(),
                "pyproject.toml" => return "Python project".to_string(),
                _ => {}
            }
        }

        // Fall back to primary language
        if let Some(primary_lang) = self.primary_languages.first() {
            format!("{} project", primary_lang)
        } else {
            "software project".to_string()
        }
    }
}

// Example usage:
//
// ```rust
// use std::path::Path;
// use cargo_chat::repo::RepoProfile;
//
// #[tokio::main]
// async fn main() -> anyhow::Result<()> {
//     let repo_path = Path::new("/path/to/repository");
//     let profile = RepoProfile::from_directory(repo_path).await?;
//
//     println!("Repository: {}", profile.name);
//     println!("Description: {}", profile.description());
//     println!("Project Type: {}", profile.project_type());
//
//     if !profile.primary_languages.is_empty() {
//         println!("Languages: {}", RepoProfile::human_list(&profile.primary_languages, "and"));
//     }
//
//     if !profile.frameworks.is_empty() {
//         println!("Frameworks: {}", RepoProfile::human_list(&profile.frameworks, "and"));
//     }
//
//     // Use with Hyde for dynamic context
//     let hyde = hyde::Hyde::new(
//         hyde_client, answer_client, embedder, ann_index, 1000, reranker, Some(profile)
//     );
//
//     Ok(())
// }
// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_human_list() {
        assert_eq!(RepoProfile::human_list(&[], "and"), "");
        assert_eq!(
            RepoProfile::human_list(&["Rust".to_string()], "and"),
            "Rust"
        );
        assert_eq!(
            RepoProfile::human_list(&["Rust".to_string(), "Python".to_string()], "and"),
            "Rust and Python"
        );
        assert_eq!(
            RepoProfile::human_list(
                &[
                    "Rust".to_string(),
                    "Python".to_string(),
                    "JavaScript".to_string()
                ],
                "and"
            ),
            "Rust, Python, and JavaScript"
        );
    }

    #[test]
    fn test_is_build_file() {
        assert!(RepoProfile::is_build_file("Cargo.toml"));
        assert!(RepoProfile::is_build_file("package.json"));
        assert!(RepoProfile::is_build_file("pom.xml"));
        assert!(RepoProfile::is_build_file("Makefile"));
        assert!(!RepoProfile::is_build_file("main.rs"));
        assert!(!RepoProfile::is_build_file("index.js"));
    }

    #[test]
    fn test_description_generation() {
        let profile = RepoProfile {
            name: "test-repo".to_string(),
            primary_languages: vec!["Rust".to_string(), "JavaScript".to_string()],
            build_files: vec!["Cargo.toml".to_string()],
            frameworks: vec!["Tokio".to_string(), "React".to_string()],
            readme_summary: None,
        };

        let desc = profile.description();
        assert!(desc.contains("test-repo"));
        assert!(desc.contains("Rust"));
        assert!(desc.contains("JavaScript"));
    }

    #[test]
    fn test_project_type_detection() {
        let rust_profile = RepoProfile {
            name: "rust-app".to_string(),
            primary_languages: vec!["Rust".to_string()],
            build_files: vec!["Cargo.toml".to_string()],
            frameworks: vec![],
            readme_summary: None,
        };
        assert_eq!(rust_profile.project_type(), "Rust project");

        let js_profile = RepoProfile {
            name: "js-app".to_string(),
            primary_languages: vec!["JavaScript".to_string()],
            build_files: vec!["package.json".to_string()],
            frameworks: vec![],
            readme_summary: None,
        };
        assert_eq!(js_profile.project_type(), "Node.js project");
    }
}
