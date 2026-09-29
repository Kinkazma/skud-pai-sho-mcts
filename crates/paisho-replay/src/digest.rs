use core::fmt;
use core::str::FromStr;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReplayDigestV1([u8; 32]);

impl ReplayDigestV1 {
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ReplayDigestV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for ReplayDigestV1 {
    type Err = ReplayDigestV1Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.len() != 64 {
            return Err(ReplayDigestV1Error::WrongLength(text.len()));
        }
        let mut bytes = [0; 32];
        for (index, pair) in text.as_bytes().chunks_exact(2).enumerate() {
            bytes[index] = decode_nibble(pair[0])?
                .checked_mul(16)
                .and_then(|high| high.checked_add(decode_nibble(pair[1]).ok()?))
                .ok_or(ReplayDigestV1Error::InvalidHex)?;
        }
        Ok(Self(bytes))
    }
}

fn decode_nibble(byte: u8) -> Result<u8, ReplayDigestV1Error> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(ReplayDigestV1Error::InvalidHex),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayDigestV1Error {
    WrongLength(usize),
    InvalidHex,
}

impl fmt::Display for ReplayDigestV1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLength(length) => {
                write!(
                    formatter,
                    "SHA-256 digest has {length} characters; expected 64"
                )
            }
            Self::InvalidHex => {
                formatter.write_str("SHA-256 digest must use lowercase hexadecimal")
            }
        }
    }
}

impl std::error::Error for ReplayDigestV1Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_hex_round_trips_canonically() {
        let digest = ReplayDigestV1::from_bytes([0xab; 32]);
        let text = digest.to_string();
        assert_eq!(text.len(), 64);
        assert_eq!(text.parse(), Ok(digest));
        assert_eq!(
            "AB".repeat(32).parse::<ReplayDigestV1>(),
            Err(ReplayDigestV1Error::InvalidHex)
        );
    }
}
