use colored::Color;
use colored::Color::TrueColor;

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::TrueColor { r, g, b }
}

pub fn html(c: Color) -> String {
    if let TrueColor { r, g, b } = c {
        format!("rgb({},{},{})", r, g, b)
    } else {
        c.to_fg_str().to_string()
    }
}

pub const KEYWORD_COLOR: Color = rgb(0, 95, 135);
pub const MACRO_COLOR: Color = rgb(95, 135, 0);
pub const VAR_COLOR: Color = rgb(215, 95, 0);
pub const VAL_COLOR: Color = rgb(68, 68, 68);
pub const TYPE_COLOR: Color = rgb(0, 135, 0);
pub const DEF_COLOR: Color = rgb(0, 135, 175);
pub const NUMERIC_COLOR: Color = rgb(215, 95, 0);
pub const STRING_COLOR: Color = rgb(95, 135, 0);
pub const BUILTIN_COLOR: Color = rgb(0, 135, 0);
pub const COMMENT_COLOR: Color = rgb(135, 135, 135);

