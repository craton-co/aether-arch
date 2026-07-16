//! Reversible x86/x86-64 branch/call/jump address normalization.
//!
//! Relative `CALL`/`JMP` operands vary with instruction position and frustrate
//! dictionary matching. The BCJ transform converts them to absolute virtual
//! positions before compression and restores the original relative values
//! after decompression. Routing still compares the encoded result with all
//! other methods, so false-positive opcode bytes cannot regress archive size.

/// Return whether `data` is an x86 or x86-64 ELF, PE, or Mach-O executable.
pub fn is_x86_executable(data: &[u8]) -> bool {
    is_x86_elf(data) || is_x86_pe(data) || is_x86_macho(data)
}

fn is_x86_elf(data: &[u8]) -> bool {
    if data.len() < 20 || &data[..4] != b"\x7FELF" {
        return false;
    }
    let machine = match data[5] {
        1 => u16::from_le_bytes([data[18], data[19]]),
        2 => u16::from_be_bytes([data[18], data[19]]),
        _ => return false,
    };
    matches!(machine, 3 | 62)
}

fn is_x86_pe(data: &[u8]) -> bool {
    if data.len() < 0x40 || &data[..2] != b"MZ" {
        return false;
    }
    let pe_offset = u32::from_le_bytes([data[0x3C], data[0x3D], data[0x3E], data[0x3F]]) as usize;
    let Some(machine_end) = pe_offset.checked_add(6) else {
        return false;
    };
    if machine_end > data.len() || &data[pe_offset..pe_offset + 4] != b"PE\0\0" {
        return false;
    }
    let machine = u16::from_le_bytes([data[pe_offset + 4], data[pe_offset + 5]]);
    matches!(machine, 0x014C | 0x8664)
}

fn is_x86_macho(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }
    let (little_endian, valid_magic) = match &data[..4] {
        [0xCE, 0xFA, 0xED, 0xFE] | [0xCF, 0xFA, 0xED, 0xFE] => (true, true),
        [0xFE, 0xED, 0xFA, 0xCE] | [0xFE, 0xED, 0xFA, 0xCF] => (false, true),
        _ => (true, false),
    };
    if !valid_magic {
        return false;
    }
    let bytes = [data[4], data[5], data[6], data[7]];
    let cpu_type = if little_endian {
        u32::from_le_bytes(bytes)
    } else {
        u32::from_be_bytes(bytes)
    };
    matches!(cpu_type, 7 | 0x0100_0007)
}

/// Apply the forward x86 BCJ transform.
pub fn encode_x86(data: &[u8]) -> Option<Vec<u8>> {
    if !is_x86_executable(data) {
        return None;
    }
    let mut transformed = data.to_vec();
    transform_x86(&mut transformed, true);
    Some(transformed)
}

/// Reverse the x86 BCJ transform in place.
pub fn decode_x86(data: &mut [u8]) {
    transform_x86(data, false);
}

fn transform_x86(data: &mut [u8], encode: bool) {
    let mut position = 0usize;
    while position.checked_add(5).is_some_and(|end| end <= data.len()) {
        if matches!(data[position], 0xE8 | 0xE9) {
            let operand = i32::from_le_bytes([
                data[position + 1],
                data[position + 2],
                data[position + 3],
                data[position + 4],
            ]);
            let instruction_end = (position as u32).wrapping_add(5) as i32;
            let normalized = if encode {
                operand.wrapping_add(instruction_end)
            } else {
                operand.wrapping_sub(instruction_end)
            };
            data[position + 1..position + 5].copy_from_slice(&normalized.to_le_bytes());
            position += 5;
        } else {
            position += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_pe() -> Vec<u8> {
        let mut data = vec![0u8; 256];
        data[..2].copy_from_slice(b"MZ");
        data[0x3C..0x40].copy_from_slice(&(0x80u32).to_le_bytes());
        data[0x80..0x84].copy_from_slice(b"PE\0\0");
        data[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        for position in (0x90..220).step_by(10) {
            data[position] = 0xE8;
            data[position + 1..position + 5]
                .copy_from_slice(&(0x1234i32 - position as i32).to_le_bytes());
        }
        data
    }

    #[test]
    fn detects_x86_pe_and_rejects_arbitrary_data() {
        assert!(is_x86_executable(&synthetic_pe()));
        assert!(!is_x86_executable(b"not an executable"));
    }

    #[test]
    fn x86_bcj_roundtrip() {
        let original = synthetic_pe();
        let mut transformed = encode_x86(&original).unwrap();
        assert_ne!(transformed, original);
        decode_x86(&mut transformed);
        assert_eq!(transformed, original);
    }
}
