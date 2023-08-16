use runtime::prelude::*;

fn main() {
    std::env::set_current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/")).unwrap();

    let model_file: File = File::open("./model/resnet18.onnx");
    let model_data: Blob = model_file.read_to_bytes();
    let model: Model = Model::new(model_data);

    let img_file: File = File::open("./data/images/cats.txt");
    let img_data: Blob = img_file.read_to_bytes();
    let img: Image = Image::new(img_data);
    let img: Image = img.resize_height(256);
    let img: Image = img.center_crop(224, 224);

    let x: Matrix<f32> = img.into_matrix().insert_axis(0);
    let y: Matrix<f32> = model.predict(x).remove_axis(0);

    let y = y.into_vec().sort();

    let labels_file: File = File::open("./models/imagenet_class_index.txt");
    let labels_data: String = labels_file.read_to_string();
    let labels: Vec<String> = labels_data.decode(Encoding::Json);

    for (i, score) in y.iter().enumerate() {
        let label = labels[i as usize].clone();
        println!("{}: {}", label, score);
    }
}
