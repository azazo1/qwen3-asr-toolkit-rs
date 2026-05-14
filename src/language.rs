use std::collections::HashMap;

use anyhow::{Result, bail};
use once_cell::sync::Lazy;

static LANGUAGE_CODE_MAPPING: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    HashMap::from([
        ("ar", "Arabic"),
        ("cs", "Czech"),
        ("da", "Danish"),
        ("de", "German"),
        ("en", "English"),
        ("es", "Spanish"),
        ("fi", "Finnish"),
        ("fil", "Filipino"),
        ("fr", "French"),
        ("hi", "Hindi"),
        ("id", "Indonesian"),
        ("is", "Icelandic"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("ms", "Malay"),
        ("no", "Norwegian"),
        ("pl", "Polish"),
        ("pt", "Portuguese"),
        ("ru", "Russian"),
        ("sv", "Swedish"),
        ("th", "Thai"),
        ("tr", "Turkish"),
        ("uk", "Ukrainian"),
        ("vi", "Vietnamese"),
        ("yue", "Cantonese"),
        ("zh", "Chinese"),
    ])
});

static LANGUAGE_ALIAS_MAPPING: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    HashMap::from([
        ("ar", "ar"),
        ("arabic", "ar"),
        ("cs", "cs"),
        ("czech", "cs"),
        ("da", "da"),
        ("danish", "da"),
        ("de", "de"),
        ("german", "de"),
        ("en", "en"),
        ("eng", "en"),
        ("english", "en"),
        ("es", "es"),
        ("spanish", "es"),
        ("fi", "fi"),
        ("finnish", "fi"),
        ("fil", "fil"),
        ("filipino", "fil"),
        ("fr", "fr"),
        ("french", "fr"),
        ("hi", "hi"),
        ("hindi", "hi"),
        ("id", "id"),
        ("indonesian", "id"),
        ("is", "is"),
        ("icelandic", "is"),
        ("it", "it"),
        ("italian", "it"),
        ("ja", "ja"),
        ("japanese", "ja"),
        ("ko", "ko"),
        ("korean", "ko"),
        ("ms", "ms"),
        ("malay", "ms"),
        ("no", "no"),
        ("norwegian", "no"),
        ("pl", "pl"),
        ("polish", "pl"),
        ("pt", "pt"),
        ("portuguese", "pt"),
        ("ru", "ru"),
        ("russian", "ru"),
        ("sv", "sv"),
        ("swedish", "sv"),
        ("th", "th"),
        ("thai", "th"),
        ("tr", "tr"),
        ("turkish", "tr"),
        ("uk", "uk"),
        ("ukrainian", "uk"),
        ("vi", "vi"),
        ("vietnamese", "vi"),
        ("yue", "yue"),
        ("cantonese", "yue"),
        ("zh", "zh"),
        ("cn", "zh"),
        ("zh-cn", "zh"),
        ("mandarin", "zh"),
        ("putonghua", "zh"),
        ("chinese", "zh"),
        ("中文", "zh"),
        ("汉语", "zh"),
        ("普通话", "zh"),
        ("粤语", "yue"),
        ("英文", "en"),
        ("英语", "en"),
        ("日语", "ja"),
        ("德语", "de"),
        ("韩语", "ko"),
        ("俄语", "ru"),
        ("法语", "fr"),
        ("葡萄牙语", "pt"),
        ("阿拉伯语", "ar"),
        ("意大利语", "it"),
        ("西班牙语", "es"),
        ("印地语", "hi"),
        ("印尼语", "id"),
        ("泰语", "th"),
        ("土耳其语", "tr"),
        ("乌克兰语", "uk"),
        ("越南语", "vi"),
        ("捷克语", "cs"),
        ("丹麦语", "da"),
        ("菲律宾语", "fil"),
        ("芬兰语", "fi"),
        ("冰岛语", "is"),
        ("马来语", "ms"),
        ("挪威语", "no"),
        ("波兰语", "pl"),
        ("瑞典语", "sv"),
    ])
});

pub fn normalize_language_code(language: Option<&str>) -> Result<Option<String>> {
    let Some(language) = language else {
        return Ok(None);
    };

    let normalized = language.trim().to_lowercase().replace('_', "-");
    if normalized.is_empty() {
        return Ok(None);
    }

    if let Some(mapped) = LANGUAGE_ALIAS_MAPPING.get(normalized.as_str()) {
        return Ok(Some((*mapped).to_string()));
    }

    let valid = regex::Regex::new(r"^[a-z]{2,3}(?:-[a-z]{2,3})?$").expect("valid regex");
    if valid.is_match(&normalized) {
        return Ok(Some(normalized));
    }

    let supported_codes = LANGUAGE_CODE_MAPPING
        .keys()
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    bail!(
        "Unsupported language '{}'. Use one of the API language codes: {}.",
        language,
        supported_codes
    )
}

pub fn code_to_language_name(code: Option<&str>, fallback: Option<&str>) -> String {
    if let Some(code) = code
        && let Some(name) = LANGUAGE_CODE_MAPPING.get(code)
    {
        return (*name).to_string();
    }
    fallback.unwrap_or("Not Supported").to_string()
}

pub fn majority_language(languages: &[String]) -> Option<String> {
    let mut counts = HashMap::<&str, usize>::new();
    for language in languages {
        *counts.entry(language.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(language, _)| language.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_language_alias() {
        assert_eq!(
            normalize_language_code(Some("中文")).expect("normalized"),
            Some("zh".to_string())
        );
    }

    #[test]
    fn normalize_language_code_passthrough() {
        assert_eq!(
            normalize_language_code(Some("en-us")).expect("normalized"),
            Some("en-us".to_string())
        );
    }
}
