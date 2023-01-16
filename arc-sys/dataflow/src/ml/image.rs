use image::io::Reader;
use image::DynamicImage;
use image::GenericImageView;
use image::ImageFormat;
use serde::Deserialize;
use serde::Deserializer;
use serde::Serialize;
use serde::Serializer;
use std::io::BufWriter;
use std::io::Cursor;
use std::rc::Rc;
use tensorflow::Tensor;

struct Image(Rc<DynamicImage>);

impl Serialize for Image {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::new();
        self.0
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Image {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        let image = Reader::new(Cursor::new(bytes))
            .with_guessed_format()
            .unwrap()
            .decode()
            .unwrap();
        Ok(Image(Rc::new(image)))
    }
}

impl Image {
    fn new(bytes: Vec<u8>) -> Image {
        let mut rd = Reader::new(Cursor::new(bytes))
            .with_guessed_format()
            .expect("Unknown image format");
        let image = rd.decode().expect("Failed to decode image");
        Image(Rc::new(image))
    }
    fn into_tensor(&self) -> Tensor<f32> {
        let vec: Vec<f32> = self
            .0
            .pixels()
            .flat_map(|(_x, _y, rgb)| [rgb[2] as f32, rgb[1] as f32, rgb[0] as f32])
            .collect();
        Tensor::new(&[self.0.height() as u64, self.0.width() as u64, 3])
            .with_values(&vec)
            .expect("Failed to create tensor")
    }
    // fn from_tensor(tensor: Tensor<f32>) {
    //     tensor
    //         .iter::<f32>()
    //         .enumerate()
    //         .for_each(|(i, v)| println!("{}: {}", i, v));
    // }
}
