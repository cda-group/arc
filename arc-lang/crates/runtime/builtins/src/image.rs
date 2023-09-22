use derive_more::Deref;
use derive_more::DerefMut;
use image::io::Reader;
use image::DynamicImage;
use image::GenericImage;
use image::GenericImageView;
use image::ImageFormat;
use ndarray::ArrayBase;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use std::io::Cursor;

use crate::array::Array;
use crate::blob::Blob;
use crate::cow::Cow;
use crate::matrix::Matrix;
use crate::path::Path;
use crate::traits::DeepClone;

#[derive(Clone, Debug)]
#[repr(C)]
pub struct Image(Cow<Inner>);

#[derive(Clone, Debug, Deref, DerefMut)]
struct Inner(DynamicImage);

impl DeepClone for DynamicImage {
    fn deep_clone(&self) -> Self {
        self.clone()
    }
}

impl DeepClone for Inner {
    fn deep_clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Serialize for Inner {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = std::vec::Vec::new();
        self.0
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Inner {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = std::vec::Vec::<u8>::deserialize(deserializer)?;
        let image = Reader::new(Cursor::new(bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        Ok(Self(image))
    }
}

impl Image {
    pub fn new(data: Blob) -> Image {
        let rd = Reader::new(Cursor::new(data.0.as_slice()))
            .with_guessed_format()
            .expect("Unknown image format");
        let img = rd.decode().expect("Failed to decode image");
        Image::from(img)
    }

    pub fn crop(self, x: u32, y: u32, new_w: u32, new_h: u32) -> Self {
        Image::from(self.0.crop_imm(x, y, new_w, new_h))
    }

    pub fn center_crop(self, new_w: u32, new_h: u32) -> Self {
        let old_w = self.0.width();
        let old_h = self.0.height();
        self.crop((old_w - new_w) / 2, (old_h - new_h) / 2, new_w, new_h)
    }

    pub fn resize(self, new_w: u32, new_h: u32) -> Self {
        Image(Cow::new(Inner(self.0.resize_exact(
            new_w,
            new_h,
            image::imageops::FilterType::Nearest,
        ))))
    }

    pub fn resize_width(self, new_w: u32) -> Self {
        let old_w = self.0.width();
        let old_h = self.0.height();
        let new_h = (old_h as f32 * (new_w as f32 / old_w as f32)) as u32;
        self.resize(new_w, new_h)
    }

    pub fn resize_height(self, new_h: u32) -> Self {
        let old_w = self.0.width();
        let old_h = self.0.height();
        let new_w = (old_w as f32 * (new_h as f32 / old_h as f32)) as u32;
        self.resize(new_w, new_h)
    }

    pub fn into_matrix(self) -> Matrix<f32> {
        let w = self.0.width() as usize;
        let h = self.0.height() as usize;
        let mut array = ArrayBase::zeros(vec![3, w, h]);
        for (x, y, rgb) in self.0.pixels() {
            let x = x as usize;
            let y = y as usize;
            array[[0, x, y]] = rgb[0] as f32;
            array[[1, x, y]] = rgb[1] as f32;
            array[[2, x, y]] = rgb[2] as f32;
            array[[3, x, y]] = rgb[3] as f32;
        }
        Matrix::from(array)
    }

    pub fn from_matrix(matrix: Matrix<f32>) -> Self {
        let w = matrix.0.shape()[1];
        let h = matrix.0.shape()[2];
        let mut img = DynamicImage::new_rgb8(w as u32, h as u32);
        for x in 0..w {
            for y in 0..h {
                img.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgba([
                        matrix.0[[0, x, y]] as u8,
                        matrix.0[[1, x, y]] as u8,
                        matrix.0[[2, x, y]] as u8,
                        matrix.0[[3, x, y]] as u8,
                    ]),
                );
            }
        }
        Image::from(img)
    }

    pub fn save(self, path: Path) {
        self.0.save(path.0).unwrap();
    }

    pub fn height(self) -> u32 {
        self.0.height()
    }

    pub fn width(self) -> u32 {
        self.0.width()
    }

    pub fn draw_box(mut self, x: u32, y: u32, w: u32, h: u32, rgba: Array<u8, 4>) {
        let rgba = rgba.0.into();
        if x + w >= self.0.width() || y + h >= self.0.height() {
            panic!("Box out of bounds");
        }
        self.0.update(|this| {
            for i in 0..w {
                this.put_pixel(x + i, y, rgba);
                this.put_pixel(x + i, y + h, rgba);
            }
            for i in 0..h {
                this.put_pixel(x, y + i, rgba);
                this.put_pixel(x + w, y + i, rgba);
            }
        })
    }
}

impl From<DynamicImage> for Image {
    fn from(image: DynamicImage) -> Self {
        Image(Cow::new(Inner(image)))
    }
}
