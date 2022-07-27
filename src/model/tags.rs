use std::path::Path;

pub fn load_tags_to_list<P: AsRef<Path>>(filepath: P) -> Vec<String> {
    let tags_string = std::fs::read_to_string(filepath).unwrap();
    tags_string.split('\n')
        .map(|tag| tag.trim().to_string())
        .collect()
}