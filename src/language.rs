use tree_sitter::Language;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SupportedLanguage {
    Rust,
    JavaScript,
    TypeScript,
    Java,
    Cpp,
    C,
    Ruby,
    CSharp,
    Swift,
    Go,
    Python,
    Markdown,
}

impl SupportedLanguage {
    /// Get the tree-sitter language for this language
    pub fn tree_sitter_language(&self) -> Language {
        match self {
            SupportedLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
            SupportedLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            SupportedLanguage::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            SupportedLanguage::Java => tree_sitter_java::LANGUAGE.into(),
            SupportedLanguage::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            SupportedLanguage::C => tree_sitter_c::LANGUAGE.into(),
            SupportedLanguage::Ruby => tree_sitter_ruby::LANGUAGE.into(),
            SupportedLanguage::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
            SupportedLanguage::Swift => tree_sitter_swift::LANGUAGE.into(),
            SupportedLanguage::Go => tree_sitter_go::LANGUAGE.into(),
            SupportedLanguage::Python => tree_sitter_python::LANGUAGE.into(),
            SupportedLanguage::Markdown => tree_sitter_md::language(),
        }
    }

    /// Get the display name for this language
    pub fn display_name(&self) -> &'static str {
        match self {
            SupportedLanguage::Rust => "Rust",
            SupportedLanguage::JavaScript => "JavaScript",
            SupportedLanguage::TypeScript => "TypeScript",
            SupportedLanguage::Java => "Java",
            SupportedLanguage::Cpp => "C++",
            SupportedLanguage::C => "C",
            SupportedLanguage::Ruby => "Ruby",
            SupportedLanguage::CSharp => "C#",
            SupportedLanguage::Swift => "Swift",
            SupportedLanguage::Go => "Go",
            SupportedLanguage::Python => "Python",
            SupportedLanguage::Markdown => "Markdown",
        }
    }

    /// Get file extensions associated with this language
    pub fn file_extensions(&self) -> &'static [&'static str] {
        match self {
            SupportedLanguage::Rust => &["rs"],
            SupportedLanguage::JavaScript => &["js", "jsx", "mjs"],
            SupportedLanguage::TypeScript => &["ts", "tsx"],
            SupportedLanguage::Java => &["java"],
            SupportedLanguage::Cpp => &["cpp", "cxx", "cc", "hpp", "hxx", "hh"],
            SupportedLanguage::C => &["c", "h"],
            SupportedLanguage::Ruby => &["rb"],
            SupportedLanguage::CSharp => &["cs"],
            SupportedLanguage::Swift => &["swift"],
            SupportedLanguage::Go => &["go"],
            SupportedLanguage::Python => &["py", "pyx", "pyi"],
            SupportedLanguage::Markdown => &["md", "markdown"],
        }
    }
}

/// Detect language from file extension
pub fn detect_language_from_extension(extension: &str) -> Option<SupportedLanguage> {
    let ext = extension.to_lowercase();
    
    for language in [
        SupportedLanguage::Rust,
        SupportedLanguage::JavaScript,
        SupportedLanguage::TypeScript,
        SupportedLanguage::Java,
        SupportedLanguage::Cpp,
        SupportedLanguage::C,
        SupportedLanguage::Ruby,
        SupportedLanguage::CSharp,
        SupportedLanguage::Swift,
        // SupportedLanguage::Kotlin, // Temporarily disabled
        SupportedLanguage::Go,
        SupportedLanguage::Python,
        SupportedLanguage::Markdown,
    ] {
        if language.file_extensions().contains(&ext.as_str()) {
            return Some(language);
        }
    }
    
    None
}

/// Get all supported file extensions
pub fn get_all_supported_extensions() -> Vec<&'static str> {
    let mut extensions = Vec::new();
    
    for language in [
        SupportedLanguage::Rust,
        SupportedLanguage::JavaScript,
        SupportedLanguage::TypeScript,
        SupportedLanguage::Java,
        SupportedLanguage::Cpp,
        SupportedLanguage::C,
        SupportedLanguage::Ruby,
        SupportedLanguage::CSharp,
        SupportedLanguage::Swift,
        // SupportedLanguage::Kotlin, // Temporarily disabled
        SupportedLanguage::Go,
        SupportedLanguage::Python,
        SupportedLanguage::Markdown,
    ] {
        extensions.extend(language.file_extensions());
    }
    
    extensions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language_from_extension() {
        assert_eq!(detect_language_from_extension("rs"), Some(SupportedLanguage::Rust));
        assert_eq!(detect_language_from_extension("js"), Some(SupportedLanguage::JavaScript));
        assert_eq!(detect_language_from_extension("ts"), Some(SupportedLanguage::TypeScript));
        assert_eq!(detect_language_from_extension("java"), Some(SupportedLanguage::Java));
        assert_eq!(detect_language_from_extension("cpp"), Some(SupportedLanguage::Cpp));
        assert_eq!(detect_language_from_extension("c"), Some(SupportedLanguage::C));
        assert_eq!(detect_language_from_extension("rb"), Some(SupportedLanguage::Ruby));
        assert_eq!(detect_language_from_extension("cs"), Some(SupportedLanguage::CSharp));
        assert_eq!(detect_language_from_extension("swift"), Some(SupportedLanguage::Swift));

        assert_eq!(detect_language_from_extension("go"), Some(SupportedLanguage::Go));
        assert_eq!(detect_language_from_extension("py"), Some(SupportedLanguage::Python));
        assert_eq!(detect_language_from_extension("md"), Some(SupportedLanguage::Markdown));
        assert_eq!(detect_language_from_extension("unknown"), None);
    }

    #[test]
    fn test_case_insensitive_detection() {
        assert_eq!(detect_language_from_extension("RS"), Some(SupportedLanguage::Rust));
        assert_eq!(detect_language_from_extension("JS"), Some(SupportedLanguage::JavaScript));
        assert_eq!(detect_language_from_extension("TS"), Some(SupportedLanguage::TypeScript));
    }
}
