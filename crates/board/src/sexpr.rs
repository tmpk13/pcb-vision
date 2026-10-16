//! A small s-expression reader for KiCad files.
//!
//! KiCad writes one top level list with nested lists and atoms. Atoms are
//! either bare tokens or double quoted strings with backslash escapes; both
//! read as the same `Atom` since the file never relies on the difference.

#[derive(Debug, Clone, PartialEq)]
pub enum Sexpr {
    Atom(String),
    List(Vec<Sexpr>),
}

#[derive(Debug, thiserror::Error)]
pub enum SexprError {
    #[error("unexpected end of input at byte {0}")]
    UnexpectedEnd(usize),
    #[error("unexpected closing parenthesis at byte {0}")]
    UnexpectedClose(usize),
    #[error("unterminated string starting at byte {0}")]
    UnterminatedString(usize),
    #[error("trailing data at byte {0}")]
    Trailing(usize),
    #[error("empty input")]
    Empty,
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn skip_space(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn read(&mut self) -> Result<Sexpr, SexprError> {
        self.skip_space();
        let Some(&c) = self.bytes.get(self.pos) else {
            return Err(SexprError::UnexpectedEnd(self.pos));
        };
        match c {
            b'(' => {
                self.pos += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_space();
                    match self.bytes.get(self.pos) {
                        None => return Err(SexprError::UnexpectedEnd(self.pos)),
                        Some(b')') => {
                            self.pos += 1;
                            return Ok(Sexpr::List(items));
                        }
                        Some(_) => items.push(self.read()?),
                    }
                }
            }
            b')' => Err(SexprError::UnexpectedClose(self.pos)),
            b'"' => {
                let start = self.pos;
                self.pos += 1;
                let mut raw: Vec<u8> = Vec::new();
                loop {
                    match self.bytes.get(self.pos) {
                        None => return Err(SexprError::UnterminatedString(start)),
                        Some(b'"') => {
                            self.pos += 1;
                            return Ok(Sexpr::Atom(String::from_utf8_lossy(&raw).into_owned()));
                        }
                        Some(b'\\') => {
                            self.pos += 1;
                            match self.bytes.get(self.pos) {
                                Some(b'n') => raw.push(b'\n'),
                                Some(b't') => raw.push(b'\t'),
                                Some(b'r') => raw.push(b'\r'),
                                Some(&e) => raw.push(e),
                                None => return Err(SexprError::UnterminatedString(start)),
                            }
                            self.pos += 1;
                        }
                        Some(&b) => {
                            raw.push(b);
                            self.pos += 1;
                        }
                    }
                }
            }
            _ => {
                let start = self.pos;
                while self.pos < self.bytes.len() {
                    let b = self.bytes[self.pos];
                    if b.is_ascii_whitespace() || b == b'(' || b == b')' || b == b'"' {
                        break;
                    }
                    self.pos += 1;
                }
                Ok(Sexpr::Atom(
                    String::from_utf8_lossy(&self.bytes[start..self.pos]).into_owned(),
                ))
            }
        }
    }
}

impl Sexpr {
    /// Read one expression; anything but whitespace after it is an error.
    pub fn parse(text: &str) -> Result<Sexpr, SexprError> {
        let mut reader = Reader {
            bytes: text.as_bytes(),
            pos: 0,
        };
        reader.skip_space();
        if reader.pos >= reader.bytes.len() {
            return Err(SexprError::Empty);
        }
        let value = reader.read()?;
        reader.skip_space();
        if reader.pos < reader.bytes.len() {
            return Err(SexprError::Trailing(reader.pos));
        }
        Ok(value)
    }

    pub fn as_atom(&self) -> Option<&str> {
        match self {
            Sexpr::Atom(s) => Some(s),
            Sexpr::List(_) => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        self.as_atom()?.parse().ok()
    }

    /// Every element of a list, the head included; empty for an atom.
    pub fn items(&self) -> &[Sexpr] {
        match self {
            Sexpr::List(v) => v,
            Sexpr::Atom(_) => &[],
        }
    }

    /// The head atom of a list.
    pub fn name(&self) -> Option<&str> {
        self.items().first()?.as_atom()
    }

    pub fn is(&self, name: &str) -> bool {
        self.name() == Some(name)
    }

    /// The elements after the head.
    pub fn args(&self) -> &[Sexpr] {
        let items = self.items();
        if items.is_empty() {
            items
        } else {
            &items[1..]
        }
    }

    pub fn arg(&self, index: usize) -> Option<&Sexpr> {
        self.args().get(index)
    }

    pub fn arg_str(&self, index: usize) -> Option<&str> {
        self.arg(index)?.as_atom()
    }

    pub fn arg_f64(&self, index: usize) -> Option<f64> {
        self.arg(index)?.as_f64()
    }

    /// The atom arguments in order, lists skipped.
    pub fn atom_args(&self) -> impl Iterator<Item = &str> {
        self.args().iter().filter_map(|a| a.as_atom())
    }

    /// The list arguments in order, atoms skipped.
    pub fn lists(&self) -> impl Iterator<Item = &Sexpr> {
        self.args().iter().filter(|a| matches!(a, Sexpr::List(_)))
    }

    /// The first child list with this head.
    pub fn find(&self, name: &str) -> Option<&Sexpr> {
        self.args().iter().find(|a| a.is(name))
    }

    /// Every child list with this head.
    pub fn find_all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Sexpr> + 'a {
        self.args().iter().filter(move |a| a.is(name))
    }

    /// `(name value)` as a number.
    pub fn find_f64(&self, name: &str) -> Option<f64> {
        self.find(name)?.arg_f64(0)
    }

    /// `(name value)` as a string.
    pub fn find_str(&self, name: &str) -> Option<&str> {
        self.find(name)?.arg_str(0)
    }

    /// `(name x y)` as a pair.
    pub fn find_xy(&self, name: &str) -> Option<(f64, f64)> {
        let list = self.find(name)?;
        Some((list.arg_f64(0)?, list.arg_f64(1)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_nested_lists_and_strings() {
        let v = Sexpr::parse(r#"(pad "1" smd rect (at 1.5 -2 90) (size 1 0.5) (net 3 "a \"b\""))"#)
            .unwrap();
        assert!(v.is("pad"));
        assert_eq!(v.arg_str(0), Some("1"));
        assert_eq!(v.arg_str(2), Some("rect"));
        assert_eq!(v.find_xy("at"), Some((1.5, -2.0)));
        assert_eq!(v.find("at").unwrap().arg_f64(2), Some(90.0));
        assert_eq!(v.find("net").unwrap().arg_str(1), Some("a \"b\""));
        assert_eq!(v.find("missing"), None);
    }

    #[test]
    fn rejects_broken_input() {
        assert!(matches!(
            Sexpr::parse("(a (b)"),
            Err(SexprError::UnexpectedEnd(_))
        ));
        assert!(matches!(
            Sexpr::parse("(a)) "),
            Err(SexprError::Trailing(_))
        ));
        assert!(matches!(Sexpr::parse("  "), Err(SexprError::Empty)));
        assert!(matches!(
            Sexpr::parse("(a \"x)"),
            Err(SexprError::UnterminatedString(_))
        ));
    }

    #[test]
    fn keeps_unicode_in_strings() {
        let v = Sexpr::parse("(t \"caf\u{e9} \u{3a9}\")").unwrap();
        assert_eq!(v.arg_str(0), Some("caf\u{e9} \u{3a9}"));
    }
}
